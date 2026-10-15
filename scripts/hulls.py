#!/usr/bin/env python3
"""
Store hull distances on one energy surface for every structure.

A structure's distance to the hull depends only on the structures in its own
chemical system and that system's subsystems, so each distinct system is
computed on its own (in parallel) and only its own structures are updated.
Run after bulk imports or relaxations; routes compute hulls on the fly and
do not depend on these stored values, but selection queries do (e.g. "PBE
structures within 100 meV/atom of the hull").

Usage:
    python scripts/hulls.py --database ggen.db --surface pbe
    python scripts/hulls.py --database ggen.db --surface pbe --source alexandria
    python scripts/hulls.py --database ggen.db --surface orb-v3-conservative-inf-mpa --system Fe-Co-Ge
"""

import argparse
import logging
import sys
import time
from collections import deque
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen.database import ON_HULL_TOLERANCE, StructureDatabase

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("ggen.database").setLevel(logging.ERROR)

_worker_db: Optional[StructureDatabase] = None


def _init_worker(path: str, surface: str) -> None:
    global _worker_db
    _worker_db = StructureDatabase(path, surface=surface, readonly=True)


def hull_batch(systems: List[str]) -> List[Tuple[str, float]]:
    """Hull distances of the structures whose own chemsys is in ``systems``."""
    db = _worker_db
    out: List[Tuple[str, float]] = []
    for chemsys in systems:
        own = {
            row[0]
            for row in db.conn.execute(
                "SELECT id FROM structures WHERE chemsys = ?", (chemsys,)
            )
        }
        try:
            _, e_hull, _ = db.compute_hull(chemsys, update_database=False)
        except Exception as exc:
            logger.warning(f"{chemsys}: {exc}")
            continue
        out.extend((sid, e) for sid, e in e_hull.items() if sid in own)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", default="ggen.db")
    parser.add_argument("--surface", required=True)
    parser.add_argument("--source", help="Only systems containing structures from this source")
    parser.add_argument("--system", help="Only this system and its subsystems")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=200)
    args = parser.parse_args()

    db = StructureDatabase(args.database, surface=args.surface)
    where = ["c.energy_per_atom IS NOT NULL"]
    params: List = [db.surface]
    if args.source:
        where.append("s.source = ?")
        params.append(args.source)
    if args.system:
        clause, subsystems = db._subsystem_clause(args.system)
        where.append(clause)
        params.extend(subsystems)
    systems = [
        row[0]
        for row in db.conn.execute(
            f"""
            SELECT DISTINCT s.chemsys FROM structures s
            JOIN calculations c ON c.structure_id = s.id AND c.surface = ?
            WHERE {' AND '.join(where)}
            """,
            params,
        )
    ]
    # Larger systems are slower; interleave so batches take similar time.
    systems.sort(key=lambda s: (s.count("-"), s))
    logger.info(f"{len(systems):,} systems on {db.surface}")

    started = time.time()
    done = updated = 0
    now = datetime.now().isoformat()
    batches = [systems[i : i + args.batch] for i in range(0, len(systems), args.batch)]

    def write(rows: List[Tuple[str, float]]) -> None:
        db.conn.executemany(
            """
            UPDATE calculations SET e_above_hull = ?, is_on_hull = ?, hull_computed_at = ?
            WHERE structure_id = ? AND surface = ?
            """,
            ((e, int(e < ON_HULL_TOLERANCE), now, sid, db.surface) for sid, e in rows),
        )
        db._commit()

    with Pool(args.workers, initializer=_init_worker, initargs=(args.database, db.surface)) as pool:
        pending = deque()
        for batch in batches:
            pending.append((len(batch), pool.apply_async(hull_batch, (batch,))))
            while len(pending) >= args.workers * 2:
                size, result = pending.popleft()
                rows = result.get()
                write(rows)
                done += size
                updated += len(rows)
                if done % (args.batch * 20) < args.batch:
                    logger.info(
                        f"  {done:,}/{len(systems):,} systems, {updated:,} structures "
                        f"({time.time() - started:.0f}s)"
                    )
        while pending:
            size, result = pending.popleft()
            rows = result.get()
            write(rows)
            done += size
            updated += len(rows)

    db.analyze()
    db.close()
    logger.info(
        f"Stored hull distances for {updated:,} structures in {len(systems):,} systems "
        f"in {time.time() - started:.0f}s"
    )


if __name__ == "__main__":
    main()
