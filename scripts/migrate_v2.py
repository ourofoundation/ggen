#!/usr/bin/env python3
"""
Migrate a version 1 ggen database to the version 2 schema.

Version 1 kept one energy (ORB) and one CIF per structure in a single wide
table. Version 2 separates identity, geometries (per slot, compressed) and
calculations (per energy surface, with MP2020 corrections); see
``ggen/database.py``. What moves where:

- GGen rows: CIF -> ORB geometry; energy and phonons -> ORB calculation.
- Materials Project rows: the original CIF kept in metadata -> source
  geometry; the ORB-relaxed CIF and energy -> ORB slots. MP's own energy is
  not carried over: it came from the summary endpoint (GGA/GGA+U/r2SCAN
  mixed), so the PBE surface is filled by re-importing MP.
- Alexandria rows: CIF -> source geometry; raw PBE energy -> PBE calculation
  (per atom of Alexandria's own cell). Alexandria's corrected energy and hull
  distance are dropped, since MP2020 is applied here instead.
- Runs and run links are copied. Hull distances are recomputed afterwards
  for the explored systems (old ones were uncorrected).

MP2020 corrections need a parsed structure per energy, so rows are prepared
in a process pool.

Usage:
    python scripts/migrate_v2.py ggen.db ggen.v2.db
    python scripts/migrate_v2.py ggen.db ggen.v2.db --workers 32
    # Cutover: bring rows written to v1 after the snapshot into v2
    python scripts/migrate_v2.py ggen.db ggen.v2.db --since 2026-10-15T12:00:00
"""

import argparse
import json
import logging
import sqlite3
import sys
import tempfile
import time
import warnings
from collections import deque
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymatgen.core import Composition, Structure

from ggen.corrections import mp2020_correction
from ggen.database import StructureDatabase, compress_cif
from ggen.surfaces import ORB_V3, PBE, SOURCE_GEOMETRY

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MIGRATED_AT = datetime.now().isoformat()

V1_COLUMNS = [
    "id", "formula", "elements", "chemsys", "stoichiometry", "energy_per_atom",
    "total_energy", "num_atoms", "space_group_number", "space_group_symbol",
    "structure_hash", "cif_content", "is_valid", "error_message",
    "generation_metadata", "source", "source_id", "is_dynamically_stable",
    "num_imaginary_modes", "min_phonon_frequency", "max_phonon_frequency",
    "phonon_supercell", "created_at", "updated_at",
]


def _parse(cif: Optional[str]) -> Optional[Structure]:
    if not cif:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return Structure.from_str(cif, fmt="cif")
    except Exception:
        return None


def _calculation(
    surface: str,
    energy_per_atom: float,
    total_energy: Optional[float],
    structure: Optional[Structure],
    row: Dict[str, Any],
    with_phonons: bool,
    now: str,
) -> Tuple:
    correction = run_type = hubbards = None
    if structure is not None:
        result = mp2020_correction(structure, energy_per_atom)
        correction, run_type = result.per_atom, result.run_type
        hubbards = json.dumps(result.hubbards)
    phonons = (
        (
            row["is_dynamically_stable"],
            row["num_imaginary_modes"],
            row["min_phonon_frequency"],
            row["max_phonon_frequency"],
            row["phonon_supercell"],
        )
        if with_phonons
        else (None,) * 5
    )
    return (
        row["id"], surface, energy_per_atom, total_energy, correction, run_type,
        hubbards, *phonons, row["created_at"], row["updated_at"],
    )


def prepare(row: Dict[str, Any]) -> Tuple[Tuple, List[Tuple], List[Tuple], bool]:
    """Turn one v1 row into (structure, geometries, calculations, unparseable)."""
    now = row["updated_at"] or row["created_at"] or MIGRATED_AT
    row["created_at"] = row["created_at"] or now
    row["updated_at"] = row["updated_at"] or now
    if not row["num_atoms"]:
        row["num_atoms"] = sum(json.loads(row["stoichiometry"]).values())
    source = row["source"] or "ggen"
    meta = json.loads(row["generation_metadata"]) if row["generation_metadata"] else {}
    source_id = row["source_id"]
    if source == "alexandria" and source_id and source_id.startswith("alexandria:"):
        source_id = source_id.split(":", 1)[1]

    cif = row["cif_content"]
    original_cif = meta.pop("mp_original_cif", None)
    alexandria_energy = meta.get("alexandria_energy")
    for key in (
        "alexandria_energy_corrected",
        "alexandria_hull_distance",
        "alexandria_formation_energy_per_atom",
    ):
        meta.pop(key, None)

    # In version 1, energy_per_atom is always the ORB energy.
    has_orb = row["energy_per_atom"] is not None
    geometries: List[Tuple] = []
    calculations: List[Tuple] = []
    unparseable = False

    if source == "ggen":
        if cif:
            geometries.append((row["id"], ORB_V3.id, compress_cif(cif)))
    elif source == "mp":
        if original_cif:
            geometries.append((row["id"], SOURCE_GEOMETRY, compress_cif(original_cif)))
        if cif:
            # Before ORB relaxation the stored CIF is MP's own geometry.
            kind = ORB_V3.id if has_orb else SOURCE_GEOMETRY
            if kind == SOURCE_GEOMETRY and original_cif:
                kind = None
            if kind:
                geometries.append((row["id"], kind, compress_cif(cif)))
    elif cif:
        # Alexandria: the stored CIF is ORB-relaxed once it has an ORB energy.
        kind = ORB_V3.id if has_orb else SOURCE_GEOMETRY
        geometries.append((row["id"], kind, compress_cif(cif)))

    if has_orb:
        structure = _parse(cif) or _parse(original_cif)
        unparseable |= structure is None
        calculations.append(
            _calculation(
                ORB_V3.id, row["energy_per_atom"], row["total_energy"], structure,
                row, True, now,
            )
        )
    if source == "alexandria" and alexandria_energy is not None:
        structure = _parse(cif)
        unparseable |= structure is None
        per_atom = alexandria_energy / row["num_atoms"]
        calculations.append(
            _calculation(PBE.id, per_atom, alexandria_energy, structure, row, False, now)
        )

    structure_row = (
        row["id"], source, source_id, row["formula"],
        Composition(row["formula"]).reduced_formula, row["elements"], row["chemsys"],
        row["stoichiometry"], row["num_atoms"], row["space_group_number"],
        row["space_group_symbol"], row["structure_hash"], row["is_valid"],
        row["error_message"], json.dumps(meta) if meta else None,
        row["created_at"], row["updated_at"],
    )
    return structure_row, geometries, calculations, unparseable


def prepare_batch(rows: List[Dict[str, Any]]):
    return [prepare(row) for row in rows]


def read_v1(path: Path, since: Optional[str], batch: int) -> Iterator[List[Dict[str, Any]]]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where = ""
    params: Tuple = ()
    if since:
        where = "WHERE created_at >= ? OR updated_at >= ?"
        params = (since, since)
    cursor = conn.execute(f"SELECT {', '.join(V1_COLUMNS)} FROM structures {where}", params)
    while True:
        rows = cursor.fetchmany(batch)
        if not rows:
            break
        yield [dict(r) for r in rows]
    conn.close()


def migrate(
    v1_path: Path, out_path: Path, workers: int, since: Optional[str], batch: int
) -> Tuple[Dict[str, int], List[str]]:
    out = StructureDatabase(out_path)
    conn = out.conn
    counts = {"structures": 0, "geometries": 0, "calculations": 0, "unparseable": 0}
    started = time.time()

    def write(prepared) -> None:
        conn.executemany(
            "INSERT OR IGNORE INTO structures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (p[0] for p in prepared),
        )
        conn.executemany(
            "INSERT OR REPLACE INTO geometries VALUES (?, ?, ?)",
            (g for p in prepared for g in p[1]),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO calculations (
                structure_id, surface, energy_per_atom, total_energy,
                correction_per_atom, run_type, hubbards, is_dynamically_stable,
                num_imaginary_modes, min_phonon_frequency, max_phonon_frequency,
                phonon_supercell, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (c for p in prepared for c in p[2]),
        )
        conn.commit()
        counts["structures"] += len(prepared)
        counts["geometries"] += sum(len(p[1]) for p in prepared)
        counts["calculations"] += sum(len(p[2]) for p in prepared)
        counts["unparseable"] += sum(1 for p in prepared if p[3])
        if counts["structures"] % (batch * 50) < batch:
            rate = counts["structures"] / (time.time() - started)
            logger.info(f"  {counts['structures']:,} structures ({rate:,.0f}/s)")

    # Pool.imap reads its whole input ahead, which here is the whole file, so
    # batches are submitted with a bounded number in flight instead.
    with Pool(workers) as pool:
        pending = deque()
        for rows in read_v1(v1_path, since, batch):
            pending.append(pool.apply_async(prepare_batch, (rows,)))
            if len(pending) >= workers * 2:
                write(pending.popleft().get())
        while pending:
            write(pending.popleft().get())

    v1 = sqlite3.connect(f"file:{v1_path}?mode=ro", uri=True)
    run_where = "WHERE started_at >= ? OR completed_at >= ?" if since else ""
    run_params = (since, since) if since else ()
    runs = v1.execute(
        f"""
        SELECT id, chemical_system, elements, parameters, started_at, completed_at,
               num_candidates, num_successful, num_failed, notes
        FROM runs {run_where}
        """,
        run_params,
    ).fetchall()
    conn.executemany(
        """
        INSERT OR REPLACE INTO runs (
            id, chemical_system, elements, parameters, started_at, completed_at,
            num_candidates, num_successful, num_failed, notes, surface
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (tuple(r) + (ORB_V3.id,) for r in runs),
    )
    link_where = (
        "WHERE run_id IN (SELECT id FROM runs WHERE started_at >= ? OR completed_at >= ?)"
        if since
        else ""
    )
    links = v1.execute(
        f"SELECT run_id, structure_id, e_above_hull, is_on_hull FROM run_structures {link_where}",
        run_params,
    ).fetchall()
    conn.executemany("INSERT OR REPLACE INTO run_structures VALUES (?, ?, ?, ?)", links)
    explored = sorted(
        {r[0] for r in v1.execute("SELECT DISTINCT chemical_system FROM runs")}
        | {r[0] for r in v1.execute("SELECT DISTINCT chemsys FROM hull_entries")}
    )
    # When each reference source was last imported, as far as v1 knows.
    for source, imported_at in v1.execute(
        "SELECT source, MAX(created_at) FROM structures WHERE source != 'ggen' GROUP BY source"
    ):
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            (f"imported:{source}", imported_at),
        )
    v1.close()
    conn.commit()
    counts["runs"] = len(runs)
    counts["run_structures"] = len(links)
    out.close()
    counts["explored_systems"] = len(explored)
    logger.info(f"Copied rows in {time.time() - started:.0f}s: {counts}")
    return counts, explored


def recompute_hulls(path: Path, systems: List[str]) -> None:
    db = StructureDatabase(path)
    started = time.time()
    for i, chemsys in enumerate(systems, 1):
        try:
            db.compute_hull(chemsys)
        except Exception as exc:
            logger.warning(f"Hull failed for {chemsys}: {exc}")
        if i % 25 == 0:
            logger.info(f"  hulls: {i}/{len(systems)}")
    db.analyze()
    db.close()
    logger.info(f"Recomputed {len(systems)} ORB hulls in {time.time() - started:.0f}s")


def report(path: Path) -> None:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    logger.info("Structures by source:")
    for row in conn.execute("SELECT source, COUNT(*) FROM structures GROUP BY source"):
        logger.info(f"  {row[0]}: {row[1]:,}")
    logger.info("Energies by surface (incompatible with MP2020 in parentheses):")
    for row in conn.execute(
        """
        SELECT surface, COUNT(*), SUM(correction_per_atom IS NULL)
        FROM calculations WHERE energy_per_atom IS NOT NULL GROUP BY surface
        """
    ):
        logger.info(f"  {row[0]}: {row[1]:,} ({row[2]:,})")
    for row in conn.execute("SELECT kind, COUNT(*) FROM geometries GROUP BY kind"):
        logger.info(f"  geometry {row[0]}: {row[1]:,}")
    conn.close()
    logger.info(f"Size: {path.stat().st_size / 2**30:.2f} GiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("v1", type=Path, help="Version 1 database (read-only)")
    parser.add_argument("v2", type=Path, help="Version 2 database to write")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument(
        "--since",
        help="Only migrate v1 rows created or updated at/after this ISO time, "
        "merging them into an existing v2 database",
    )
    parser.add_argument("--skip-hulls", action="store_true")
    args = parser.parse_args()

    if args.since:
        if not args.v2.exists():
            parser.error("--since merges into an existing v2 database")
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "delta.db"
            counts, _ = migrate(args.v1, shard, args.workers, args.since, args.batch)
            db = StructureDatabase(args.v2)
            db.merge_from(shard, recompute_hulls=not args.skip_hulls)
            db.close()
        report(args.v2)
        return

    if args.v2.exists():
        parser.error(f"{args.v2} exists; migrate into a fresh file")
    counts, explored = migrate(args.v1, args.v2, args.workers, None, args.batch)
    if not args.skip_hulls:
        recompute_hulls(args.v2, explored)
    report(args.v2)


if __name__ == "__main__":
    main()
