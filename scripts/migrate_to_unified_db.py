#!/usr/bin/env python3
"""
Migrate existing exploration runs to unified database.

This script imports all structures from existing exploration run directories
into the unified ggen database, enabling cross-system structure sharing and
global hull tracking.

Usage:
    python scripts/migrate_to_unified_db.py [exploration_runs_dir] [--db-path PATH]

Example:
    python scripts/migrate_to_unified_db.py ./exploration_runs
    python scripts/migrate_to_unified_db.py ./exploration_runs --db-path ~/.ggen/ggen.db
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen import StructureDatabase


def main():
    parser = argparse.ArgumentParser(
        description="Migrate exploration runs to unified database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source_directory",
        type=str,
        nargs="?",
        default="./exploration_runs",
        help="Directory containing exploration run folders (default: ./exploration_runs)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to unified database (default: ./ggen.db in current directory)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="exploration_*",
        help="Glob pattern for run directories (default: exploration_*)",
    )
    parser.add_argument(
        "--no-recompute-hulls",
        action="store_true",
        help="Skip hull recomputation after import",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Validate source directory
    source_dir = Path(args.source_directory)
    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        sys.exit(1)

    # Initialize database
    db = StructureDatabase(args.db_path)
    logger.info(f"Using database: {db.db_path}")

    # Show pre-migration stats
    pre_stats = db.get_statistics()
    logger.info(
        f"Pre-migration: {pre_stats['total_structures']} structures, "
        f"{pre_stats['unique_formulas']} unique formulas"
    )

    # Run migration
    logger.info(f"Importing runs from: {source_dir}")
    results = db.import_all_runs(
        base_directory=source_dir,
        pattern=args.pattern,
        recompute_hulls=not args.no_recompute_hulls,
    )

    # Show results
    print("\n" + "=" * 60)
    print("MIGRATION RESULTS")
    print("=" * 60)

    total_imported = 0
    total_skipped = 0
    for run_path, (imported, skipped) in sorted(results.items()):
        run_name = Path(run_path).name
        total_imported += imported
        total_skipped += skipped
        if imported > 0 or skipped > 0:
            print(f"  {run_name}: {imported} imported, {skipped} skipped")

    print("-" * 60)
    print(f"Total: {total_imported} structures imported, {total_skipped} skipped")
    print(f"From {len(results)} exploration runs")

    # Show post-migration stats
    post_stats = db.get_statistics()
    print("\n" + "=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total structures: {post_stats['total_structures']}")
    print(f"Valid structures: {post_stats['valid_structures']}")
    print(f"Unique formulas: {post_stats['unique_formulas']}")
    print(f"Chemical systems: {post_stats['unique_chemsys']}")
    print(f"Exploration runs: {post_stats['total_runs']}")
    print(f"\nExplored systems: {', '.join(post_stats['explored_systems'])}")

    db.close()
    print(f"\nDatabase saved to: {db.db_path}")


if __name__ == "__main__":
    main()
