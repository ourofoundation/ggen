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
import logging
import os
import sys
import time
import warnings
from collections import deque
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen import Colors, StructureDatabase
from ggen.sources import LICENSES, SourceRecord, alexandria, chunk_records, materials_project
from ggen.surfaces import PBE, SOURCE_GEOMETRY

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _space_group_symbol(number: int) -> Optional[str]:
    from pymatgen.symmetry.groups import SpaceGroup

    try:
        return SpaceGroup.from_int_number(int(number)).symbol
    except Exception:
        return None


def prepare(record: SourceRecord) -> Optional[Dict[str, Any]]:
    """Everything a record needs written, computed off the main process."""
    from ggen.corrections import mp2020_correction

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

    return {
        "source_id": record.source_id,
        "formula": composition.reduced_formula,
        "stoichiometry": {
            el.symbol: int(round(n)) for el, n in composition.element_composition.items()
        },
        "num_atoms": len(structure),
        "space_group_number": sg_number,
        "space_group_symbol": sg_symbol,
        "cif": cif,
        "energy_per_atom": record.energy_per_atom,
        "correction_per_atom": correction,
        "parameters": {"run_type": run_type, "hubbards": hubbards or {}},
        "metadata": {**record.metadata, "license": LICENSES.get(record.source)},
    }


def prepare_batch(records: List[SourceRecord]) -> List[Dict[str, Any]]:
    return [p for p in (prepare(r) for r in records) if p is not None]


def write(db: StructureDatabase, source: str, existing: Dict[str, str], rows) -> Dict[str, int]:
    counts = {"new": 0, "updated": 0}
    for row in rows:
        sid = existing.get(row["source_id"])
        if sid is None:
            sid = db.add_structure(
                formula=row["formula"],
                stoichiometry=row["stoichiometry"],
                energy_per_atom=None,
                total_energy=None,
                num_atoms=row["num_atoms"],
                space_group_number=row["space_group_number"],
                space_group_symbol=row["space_group_symbol"],
                cif_content=row["cif"],
                generation_metadata=row["metadata"],
                source=source,
                source_id=row["source_id"],
                geometry_kind=SOURCE_GEOMETRY,
                check_existing=False,
                defer_commit=True,
            )
            existing[row["source_id"]] = sid
            counts["new"] += 1
        else:
            db.put_geometry(sid, SOURCE_GEOMETRY, row["cif"], defer_commit=True)
            db.update_structure_space_group(
                sid,
                space_group_number=row["space_group_number"],
                space_group_symbol=row["space_group_symbol"],
                defer_commit=True,
            )
            counts["updated"] += 1
        if row["energy_per_atom"] is not None:
            db.set_calculation(
                sid,
                row["energy_per_atom"],
                surface=PBE.id,
                total_energy=row["energy_per_atom"] * row["num_atoms"],
                correction_per_atom=row["correction_per_atom"],
                compute_correction=False,
                parameters=row["parameters"],
                defer_commit=True,
            )
    db._commit()
    return counts


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

    totals = {"new": 0, "updated": 0}
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

        for batch in chunk_records(records, args.batch):
            pending.append(pool.apply_async(prepare_batch, (batch,)))
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
        f"{C.GREEN}Done:{C.RESET} {totals['new']:,} new, {totals['updated']:,} refreshed "
        f"in {time.time() - started:.0f}s"
    )
    logger.info(f"{C.CYAN}Next:{C.RESET} python scripts/hulls.py --database {args.database} --surface pbe")


if __name__ == "__main__":
    main()
