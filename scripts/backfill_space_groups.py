#!/usr/bin/env python3
"""
Backfill missing space group number/symbol in the ggen database.

Two modes:
  1. Symbol from number: rows that have space_group_number but NULL
     space_group_symbol (e.g. Alexandria import with spg as int) are updated
     using pymatgen's space group table.
  2. Spglib from CIF: rows with both NULL are updated by parsing the CIF and
     running spglib/SpacegroupAnalyzer to detect symmetry.

Usage:
    python scripts/backfill_space_groups.py                    # Both modes
    python scripts/backfill_space_groups.py --symbol-only       # Only derive symbol from number
    python scripts/backfill_space_groups.py --spglib-only      # Only run spglib on NULL rows
    python scripts/backfill_space_groups.py --source alexandria # Limit to one source
    python scripts/backfill_space_groups.py --dry-run          # Report counts only
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen import Colors, StructureDatabase

# Cache: space group number (1–230) → symbol. Built once to avoid repeated pymatgen lookups.
_SG_NUMBER_TO_SYMBOL_CACHE: dict[int, str] | None = None


def _get_sg_number_to_symbol_cache() -> dict[int, str]:
    """Build and return a cache of space group number → symbol for 1–230."""
    global _SG_NUMBER_TO_SYMBOL_CACHE
    if _SG_NUMBER_TO_SYMBOL_CACHE is not None:
        return _SG_NUMBER_TO_SYMBOL_CACHE
    from pymatgen.symmetry.groups import SpaceGroup

    _SG_NUMBER_TO_SYMBOL_CACHE = {}
    for n in range(1, 231):
        try:
            _SG_NUMBER_TO_SYMBOL_CACHE[n] = SpaceGroup.from_int_number(n).symbol
        except Exception:
            pass
    return _SG_NUMBER_TO_SYMBOL_CACHE


def backfill_symbol_from_number(
    db: StructureDatabase,
    source: str | None = None,
    dry_run: bool = False,
    batch_size: int = 5000,
) -> int:
    """
    Update space_group_symbol from space_group_number where symbol is NULL.

    Returns number of structures updated.
    """
    cache = _get_sg_number_to_symbol_cache()
    conditions = ["space_group_number IS NOT NULL", "space_group_symbol IS NULL"]
    params = []
    if source:
        conditions.append("source = ?")
        params.append(source)

    query = "SELECT id, space_group_number FROM structures WHERE " + " AND ".join(conditions)
    rows = db.conn.execute(query, params).fetchall()
    to_update = []
    for row in rows:
        sg_num = row["space_group_number"]
        if sg_num is None or sg_num < 1 or sg_num > 230:
            continue
        sym = cache.get(int(sg_num))
        if sym:
            to_update.append((row["id"], sym))

    if dry_run:
        return len(to_update)

    now = datetime.now().isoformat()
    for i in range(0, len(to_update), batch_size):
        batch = to_update[i : i + batch_size]
        db.conn.executemany(
            "UPDATE structures SET space_group_symbol = ?, updated_at = ? WHERE id = ?",
            [(symbol, now, structure_id) for structure_id, symbol in batch],
        )
        db._commit()
    return len(to_update)


def backfill_spglib(
    db: StructureDatabase,
    source: str | None = None,
    dry_run: bool = False,
    symprec: float = 0.1,
    batch_size: int = 500,
) -> int:
    """
    For structures with both space_group_number and space_group_symbol NULL,
    parse CIF, run SpacegroupAnalyzer, and update both columns.

    Returns number of structures updated.
    """
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from tqdm import tqdm

    conditions = [
        "space_group_number IS NULL",
        "space_group_symbol IS NULL",
        "cif_content IS NOT NULL",
        "cif_content != ''",
        "is_valid = 1",
    ]
    params = []
    if source:
        conditions.append("source = ?")
        params.append(source)

    query = "SELECT id, cif_content FROM structures WHERE " + " AND ".join(conditions)
    rows = db.conn.execute(query, params).fetchall()

    if dry_run:
        return len(rows)

    updated = 0
    for i, row in enumerate(tqdm(rows, desc="Spglib", unit="struct")):
        structure_id = row["id"]
        cif_content = row["cif_content"]
        if not cif_content:
            continue
        try:
            structure = Structure.from_str(cif_content, fmt="cif")
            analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
            sg_number = analyzer.get_space_group_number()
            sg_symbol = analyzer.get_space_group_symbol()
            db.update_structure_space_group(
                structure_id,
                space_group_number=sg_number,
                space_group_symbol=sg_symbol,
                defer_commit=True,
            )
            updated += 1
        except Exception:
            continue
        if (updated % batch_size) == 0 and updated > 0:
            db._commit()
    if updated > 0:
        db._commit()
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing space group data in the ggen database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        "-d",
        default="ggen.db",
        help="Path to ggen database (default: ggen.db)",
    )
    parser.add_argument(
        "--symbol-only",
        action="store_true",
        help="Only backfill symbol from existing space_group_number",
    )
    parser.add_argument(
        "--spglib-only",
        action="store_true",
        help="Only backfill using spglib/SpacegroupAnalyzer on CIF (where both are NULL)",
    )
    parser.add_argument(
        "--source",
        choices=["ggen", "mp", "alexandria"],
        help="Limit backfill to this source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many rows would be updated",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    db_path = Path(args.db)
    if not db_path.exists():
        print(
            f"{Colors.RED}Error:{Colors.RESET} Database not found: {db_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    db = StructureDatabase(str(db_path))
    C = Colors

    try:
        do_symbol = not args.spglib_only
        do_spglib = not args.symbol_only

        if args.dry_run:
            print(f"{C.BOLD}Dry run — no changes will be written.{C.RESET}\n")

        if do_symbol:
            n = backfill_symbol_from_number(
                db, source=args.source, dry_run=args.dry_run
            )
            label = "would update" if args.dry_run else "updated"
            print(f"  Symbol from number: {C.CYAN}{n}{C.RESET} structures {label}")

        if do_spglib:
            n = backfill_spglib(
                db, source=args.source, dry_run=args.dry_run
            )
            label = "would update" if args.dry_run else "updated"
            print(f"  Spglib from CIF:    {C.CYAN}{n}{C.RESET} structures {label}")

        if args.dry_run:
            print(f"\n{C.DIM}Run without --dry-run to apply changes.{C.RESET}")
        else:
            print(f"\n{C.GREEN}Done.{C.RESET}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
