#!/usr/bin/env python3
"""
Import reference structures and their PBE energies into the ggen database.

Supports importing from:
  - Materials Project (provider='mp'): GGA/GGA+U entries, MP's own MP2020
    corrections
  - Alexandria local .json.bz2 chunks (provider='alexandria'): raw PBE
    energies, MP2020 applied here

Each structure is stored as published (the ``source`` geometry) with a
``pbe`` calculation; see ``ggen/sources.py``. Re-running an import refreshes
existing rows in place, so it doubles as an update. Structures still need
MLIP energies for MLIP hulls: run relax.py afterwards, e.g. on the ones
within 100 meV/atom of the PBE hull:

    python scripts/hulls.py --database ggen.db --surface pbe
    python scripts/relax.py --database ggen.db --unrelaxed \\
        --hull-surface pbe --e-above-hull 0.1

Usage:
    # Materials Project (MP_API_KEY in the environment or --api-key)
    python import.py --database ggen.db --provider mp

    # Alexandria from downloaded chunk files
    python import.py --database ggen.db --provider alexandria \\
        --alexandria-local-dir alexandria_pbe_2025_07_02

    # Filter to materials containing these elements
    python import.py --database ggen.db --provider mp --elements Fe Co

    # Dry run: count what would be imported
    python import.py --database ggen.db --provider alexandria \\
        --alexandria-local-dir ... --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
import warnings
from collections import deque
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen import Colors, StructureDatabase
from ggen.database import compress_cif
from ggen.sources import (
    LICENSES,
    NON_COMMERCIAL_LICENSES,
    SourceRecord,
    alexandria,
    alexandria_chunk,
    alexandria_chunks,
    chunk_records,
    materials_project,
)
from ggen.surfaces import PBE, SOURCE_GEOMETRY

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _space_group_symbol(number: int) -> Optional[str]:
    # Building a SpaceGroup costs ~30 ms, most of an import's time uncached.
    from pymatgen.symmetry.groups import SpaceGroup

    try:
        return SpaceGroup.from_int_number(int(number)).symbol
    except Exception:
        return None


_NOW = time.strftime("%Y-%m-%dT%H:%M:%S")


def prepare(record: SourceRecord, source: str) -> Optional[Dict[str, Any]]:
    """A record as ready-to-insert values, computed off the main process."""
    from ggen.corrections import mp2020_correction

    if record.license in NON_COMMERCIAL_LICENSES:
        return {"source_id": record.source_id, "excluded_license": record.license}

    structure = record.structure
    composition = structure.composition
    try:
        cif = structure.to(fmt="cif")  # unsymmetrized: the cell the energy is for
    except Exception:
        return None

    sg_number, sg_symbol = record.space_group_number, record.space_group_symbol
    if sg_number is None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number = analyzer.get_space_group_number()
            sg_symbol = analyzer.get_space_group_symbol()
        except Exception:
            pass
    if sg_number is not None and sg_symbol is None:
        sg_symbol = _space_group_symbol(sg_number)

    correction = record.correction_per_atom
    run_type = (record.parameters or {}).get("run_type")
    hubbards = (record.parameters or {}).get("hubbards")
    if record.energy_per_atom is not None and correction is None:
        result = mp2020_correction(structure, record.energy_per_atom, record.parameters)
        correction, run_type, hubbards = result.per_atom, result.run_type, result.hubbards

    stoichiometry = {
        el.symbol: int(round(n)) for el, n in composition.element_composition.items()
    }
    elements = sorted(stoichiometry)
    formula = composition.reduced_formula
    return {
        "source_id": record.source_id,
        "formula": formula,
        "reduced_formula": formula,
        "elements": json.dumps(elements),
        "chemsys": "-".join(elements),
        "stoichiometry": json.dumps(stoichiometry),
        "num_atoms": len(structure),
        "space_group_number": sg_number,
        "space_group_symbol": sg_symbol,
        "structure_hash": StructureDatabase.compute_structure_hash(
            formula, 0.0, sg_number, source, record.source_id
        ),
        "metadata": json.dumps({**record.metadata, "license": LICENSES.get(record.source)}),
        "cif": compress_cif(cif),
        "energy_per_atom": record.energy_per_atom,
        "correction_per_atom": correction,
        "run_type": run_type,
        "hubbards": json.dumps(hubbards or {}),
    }


def prepare_batch(records: List[SourceRecord], source: str) -> List[Dict[str, Any]]:
    return [p for p in (prepare(r, source) for r in records) if p is not None]


def prepare_alexandria_chunk(path: str, elements: Optional[List[str]]) -> List[Dict[str, Any]]:
    """One Alexandria chunk, read, parsed and prepared in a worker."""
    return prepare_batch(list(alexandria_chunk(Path(path), elements)), "alexandria")


def write(db: StructureDatabase, source: str, existing: Dict[str, str], rows) -> Dict[str, int]:
    """Insert prepared rows in bulk: new structures, refreshed ones, retired ones."""
    conn = db.conn
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    retired, new, refreshed = [], [], []
    for row in rows:
        sid = existing.get(row["source_id"])
        if "excluded_license" in row:
            if sid is not None:
                retired.append((f"license {row['excluded_license']}", now, sid))
            continue
        if sid is None:
            sid = str(uuid.uuid4())
            existing[row["source_id"]] = sid
            new.append((sid, row))
        else:
            refreshed.append((sid, row))

    conn.executemany(
        "UPDATE structures SET is_valid = 0, error_message = ?, updated_at = ? WHERE id = ?",
        retired,
    )
    conn.executemany(
        """
        INSERT INTO structures (
            id, source, source_id, formula, reduced_formula, elements, chemsys,
            stoichiometry, num_atoms, space_group_number, space_group_symbol,
            structure_hash, is_valid, error_message, metadata, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?, ?)
        """,
        (
            (
                sid, source, r["source_id"], r["formula"], r["reduced_formula"],
                r["elements"], r["chemsys"], r["stoichiometry"], r["num_atoms"],
                r["space_group_number"], r["space_group_symbol"], r["structure_hash"],
                r["metadata"], now, now,
            )
            for sid, r in new
        ),
    )
    conn.executemany(
        """
        UPDATE structures SET space_group_number = COALESCE(?, space_group_number),
            space_group_symbol = COALESCE(?, space_group_symbol), updated_at = ?
        WHERE id = ?
        """,
        ((r["space_group_number"], r["space_group_symbol"], now, sid) for sid, r in refreshed),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO geometries (structure_id, kind, cif) VALUES (?, ?, ?)",
        ((sid, SOURCE_GEOMETRY, r["cif"]) for sid, r in new + refreshed),
    )
    conn.executemany(
        """
        INSERT INTO calculations (
            structure_id, surface, energy_per_atom, total_energy, correction_per_atom,
            run_type, hubbards, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (structure_id, surface) DO UPDATE SET
            energy_per_atom = excluded.energy_per_atom,
            total_energy = excluded.total_energy,
            correction_per_atom = excluded.correction_per_atom,
            run_type = excluded.run_type,
            hubbards = excluded.hubbards,
            e_above_hull = NULL, is_on_hull = NULL, hull_computed_at = NULL,
            updated_at = excluded.updated_at
        """,
        (
            (
                sid, PBE.id, r["energy_per_atom"], r["energy_per_atom"] * r["num_atoms"],
                r["correction_per_atom"], r["run_type"], r["hubbards"], now, now,
            )
            for sid, r in new + refreshed
            if r["energy_per_atom"] is not None
        ),
    )
    conn.commit()
    return {"new": len(new), "updated": len(refreshed), "excluded": len(retired)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", type=str, default="ggen.db")
    parser.add_argument("--provider", choices=["mp", "alexandria"], default="mp")
    parser.add_argument("--api-key", type=str, default=None, help="Materials Project API key")
    parser.add_argument("--alexandria-local-dir", type=str, default=None)
    parser.add_argument("--alexandria-local-pattern", type=str, default="alexandria_*.json.bz2")
    parser.add_argument("--elements", type=str, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    C = Colors
    source = args.provider
    if source == "mp":
        api_key = args.api_key or os.getenv("MP_API_KEY")
        if not api_key:
            parser.error("Set MP_API_KEY or pass --api-key")
        records = materials_project(api_key, elements=args.elements)
    else:
        if not args.alexandria_local_dir:
            parser.error("provider=alexandria requires --alexandria-local-dir")
        records = alexandria(
            Path(args.alexandria_local_dir),
            pattern=args.alexandria_local_pattern,
            elements=args.elements,
        )

    db = StructureDatabase(args.database)
    existing = {
        row[0]: row[1]
        for row in db.conn.execute(
            "SELECT source_id, id FROM structures WHERE source = ? AND source_id IS NOT NULL",
            (source,),
        )
    }
    logger.info(f"{C.BOLD}{source} import{C.RESET} into {args.database}")
    logger.info(f"  {len(existing):,} {source} structures already in the database")

    if args.dry_run:
        seen = new = 0
        for record in records:
            seen += 1
            new += record.source_id not in existing
        logger.info(f"  {seen:,} records, {new:,} new, {seen - new:,} would be refreshed")
        db.close()
        return

    totals = {"new": 0, "updated": 0, "excluded": 0}
    started = time.time()
    with Pool(args.workers) as pool:
        pending = deque()

        def drain_one():
            counts = write(db, source, existing, pending.popleft().get())
            for key in totals:
                totals[key] += counts[key]
            done = totals["new"] + totals["updated"]
            logger.info(
                f"  {done:,} written ({totals['new']:,} new) "
                f"at {done / (time.time() - started):,.0f}/s"
            )

        if source == "alexandria":
            # Whole chunk files per worker: reading and parsing them is most
            # of the work, so it must not happen in the main process.
            for path in alexandria_chunks(Path(args.alexandria_local_dir), args.alexandria_local_pattern):
                pending.append(
                    pool.apply_async(prepare_alexandria_chunk, (str(path), args.elements))
                )
                if len(pending) >= args.workers:
                    drain_one()
        else:
            for batch in chunk_records(records, args.batch):
                pending.append(pool.apply_async(prepare_batch, (batch, source)))
                if len(pending) >= args.workers * 2:
                    drain_one()
        while pending:
            drain_one()

    db.conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (f"imported:{source}", time.strftime("%Y-%m-%dT%H:%M:%S")),
    )
    db.analyze()
    db.close()
    logger.info("")
    logger.info(
        f"{C.GREEN}Done:{C.RESET} {totals['new']:,} new, {totals['updated']:,} refreshed, "
        f"{totals['excluded']:,} left out for their license "
        f"in {time.time() - started:.0f}s"
    )
    logger.info(f"{C.CYAN}Next:{C.RESET} python scripts/hulls.py --database {args.database} --surface pbe")


if __name__ == "__main__":
    main()
