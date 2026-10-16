#!/usr/bin/env python3
"""
Store the sources' own published hull distances as PBE hull distances.

Computing PBE hulls over every system (scripts/hulls.py) takes about an
hour; when the distances only select structures (e.g. which to relax near
the hull) the published ones are close enough. Each is measured against its
own source's hull and corrections: Materials Project's GGA/GGA+U hull from
the import's metadata, Alexandria's from its chunk files. ``hull_computed_at``
is set to ``published`` so they can be told apart from computed ones.

Usage:
    python scripts/published_hulls.py --database ggen.db \\
        --alexandria-local-dir alexandria_pbe_2025_07_02
"""

import argparse
import bz2
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen.database import ON_HULL_TOLERANCE, StructureDatabase
from ggen.sources import alexandria_chunks
from ggen.surfaces import PBE

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def chunk_distances(path: str) -> List[Tuple[str, float]]:
    with bz2.open(path, "rt") as handle:
        payload = json.load(handle)
    out = []
    for entry in payload.get("entries", []):
        data = entry.get("data") or {}
        material_id = data.get("mat_id") or entry.get("entry_id")
        e_hull = data.get("e_above_hull")
        if material_id and e_hull is not None:
            out.append((str(material_id), float(e_hull)))
    return out


def store(db: StructureDatabase, source: str, distances) -> int:
    conn = db.conn
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _published (source_id TEXT PRIMARY KEY, e REAL)")
    conn.execute("DELETE FROM temp._published")
    conn.executemany("INSERT OR REPLACE INTO temp._published VALUES (?, ?)", distances)
    cursor = conn.execute(
        """
        UPDATE calculations AS c
        SET e_above_hull = max(p.e, 0.0), is_on_hull = (p.e < ?), hull_computed_at = 'published'
        FROM structures s JOIN temp._published p ON p.source_id = s.source_id
        WHERE c.structure_id = s.id AND c.surface = ? AND s.source = ?
        """,
        (ON_HULL_TOLERANCE, PBE.id, source),
    )
    conn.commit()
    return cursor.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", default="ggen.db")
    parser.add_argument("--alexandria-local-dir", default=None)
    parser.add_argument("--alexandria-local-pattern", default="alexandria_*.json.bz2")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    started = time.time()
    db = StructureDatabase(args.database, surface=PBE.id)

    mp = [
        (row[0], row[1])
        for row in db.conn.execute(
            """
            SELECT source_id, json_extract(metadata, '$.mp_energy_above_hull')
            FROM structures
            WHERE source = 'mp' AND json_extract(metadata, '$.mp_energy_above_hull') IS NOT NULL
            """
        )
    ]
    logger.info(f"Materials Project: {store(db, 'mp', mp):,} distances stored")

    if args.alexandria_local_dir:
        chunks = alexandria_chunks(Path(args.alexandria_local_dir), args.alexandria_local_pattern)
        total = 0
        with Pool(args.workers) as pool:
            for distances in pool.imap_unordered(chunk_distances, [str(c) for c in chunks]):
                total += store(db, "alexandria", distances)
        logger.info(f"Alexandria: {total:,} distances stored from {len(chunks)} chunks")

    near = db.conn.execute(
        "SELECT COUNT(*) FROM calculations WHERE surface = ? AND e_above_hull <= 0.1",
        (PBE.id,),
    ).fetchone()[0]
    db.analyze()
    db.close()
    logger.info(f"{near:,} PBE structures within 100 meV/atom; {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
