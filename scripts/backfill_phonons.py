#!/usr/bin/env python3
"""
Backfill phonon (dynamical stability) calculations for existing database entries.

Focuses on low-energy entries near the convex hull that don't have phonon data yet.

Usage:
    # Backfill all systems, entries within 100 meV of hull
    python backfill_phonons.py

    # Specific chemical system
    python backfill_phonons.py --system Fe-Mn-Co

    # Only entries on the hull (0 meV)
    python backfill_phonons.py --e-above-hull 0.0

    # Limit number of structures to process
    python backfill_phonons.py --max-structures 10
"""

import argparse
import logging
import sys
import warnings
from typing import List, Optional, Tuple

from ggen import Colors, StructureDatabase
from ggen.database import StoredStructure
from ggen.phonons import calculate_phonons

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=UserWarning, module="orb_models")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_entries_needing_phonons(
    db: StructureDatabase,
    chemical_system: Optional[str] = None,
    e_above_hull_cutoff: float = 0.1,
    max_structures: Optional[int] = None,
) -> List[StoredStructure]:
    """
    Get database entries that need phonon calculations.

    Args:
        db: Database connection
        chemical_system: Optional specific system (e.g., "Fe-Mn-Co"). If None, all systems.
        e_above_hull_cutoff: Maximum energy above hull in eV/atom
        max_structures: Optional limit on number of structures

    Returns:
        List of StoredStructure objects needing phonon calculations
    """
    conn = db.conn

    if chemical_system:
        # Get entries for specific system
        chemsys = db.normalize_chemsys(chemical_system)
        query = """
            SELECT s.*, h.e_above_hull, h.is_on_hull
            FROM structures s
            JOIN hull_entries h ON s.id = h.structure_id
            WHERE h.chemsys = ?
              AND h.e_above_hull <= ?
              AND s.is_dynamically_stable IS NULL
              AND s.cif_content IS NOT NULL
            ORDER BY h.e_above_hull ASC, s.energy_per_atom ASC
        """
        params: Tuple = (chemsys, e_above_hull_cutoff)
    else:
        # Get entries across all systems
        query = """
            SELECT DISTINCT s.*, MIN(h.e_above_hull) as e_above_hull, MAX(h.is_on_hull) as is_on_hull
            FROM structures s
            JOIN hull_entries h ON s.id = h.structure_id
            WHERE h.e_above_hull <= ?
              AND s.is_dynamically_stable IS NULL
              AND s.cif_content IS NOT NULL
            GROUP BY s.id
            ORDER BY e_above_hull ASC, s.energy_per_atom ASC
        """
        params = (e_above_hull_cutoff,)

    if max_structures:
        query += f" LIMIT {max_structures}"

    rows = conn.execute(query, params).fetchall()

    structures = []
    for row in rows:
        s = db._row_to_structure(row)
        s.e_above_hull = row["e_above_hull"]
        s.is_on_hull = bool(row["is_on_hull"])
        structures.append(s)

    return structures


def run_phonon_calculation(
    structure: StoredStructure,
    calculator,
    supercell: Tuple[int, int, int] = (2, 2, 2),
) -> dict:
    """
    Run phonon calculation for a structure.

    Returns dict with phonon results or error info.
    """
    pymatgen_structure = structure.get_structure()
    if pymatgen_structure is None:
        return {"error": "Could not load structure from CIF"}

    try:
        result = calculate_phonons(
            structure=pymatgen_structure,
            calculator=calculator,
            supercell=supercell,
            generate_plot=False,
        )
        return {
            "is_stable": result.is_stable,
            "num_imaginary_modes": result.num_imaginary_modes,
            "min_frequency": result.min_frequency,
            "max_frequency": result.max_frequency,
            "supercell": supercell,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Backfill phonon calculations for database entries"
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Chemical system to process (e.g., 'Fe-Mn-Co'). Default: all systems.",
    )
    parser.add_argument(
        "--e-above-hull",
        type=float,
        default=0.1,
        help="Max energy above hull in eV/atom (default: 0.1 = 100 meV)",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Maximum number of structures to process",
    )
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=[2, 2, 2],
        metavar=("X", "Y", "Z"),
        help="Supercell dimensions for phonon calculation (default: 2 2 2)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="ggen.db",
        help="Path to database file (default: ggen.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List structures that would be processed without running calculations",
    )
    args = parser.parse_args()

    C = Colors

    # Connect to database
    logger.info(f"{C.BOLD}Phonon Backfill{C.RESET}")
    logger.info(f"{C.DIM}{'=' * 50}{C.RESET}")
    logger.info(f"Database: {C.CYAN}{args.database}{C.RESET}")

    try:
        db = StructureDatabase(args.database)
    except Exception as e:
        logger.error(f"{C.RED}Failed to open database: {e}{C.RESET}")
        sys.exit(1)

    # Get structures needing phonon calculations
    logger.info(
        f"Finding entries with E_hull ≤ {args.e_above_hull * 1000:.0f} meV "
        f"missing phonon data..."
    )

    entries = get_entries_needing_phonons(
        db=db,
        chemical_system=args.system,
        e_above_hull_cutoff=args.e_above_hull,
        max_structures=args.max_structures,
    )

    if not entries:
        logger.info(f"{C.GREEN}No entries need phonon calculations!{C.RESET}")
        db.close()
        return

    logger.info(f"Found {C.YELLOW}{len(entries)}{C.RESET} entries to process")
    logger.info("")

    if args.dry_run:
        logger.info(f"{C.BOLD}Dry run - would process:{C.RESET}")
        for i, s in enumerate(entries):
            e_hull = s.e_above_hull or 0
            logger.info(
                f"  {i + 1:3d}. {s.formula:12s}  "
                f"E={s.energy_per_atom:.4f} eV/atom  "
                f"SG={s.space_group_symbol:10s}  "
                f"E_hull={e_hull * 1000:.1f} meV"
            )
        db.close()
        return

    # Initialize calculator
    logger.info(f"{C.BOLD}Initializing calculator...{C.RESET}")
    try:
        from ggen.calculator import get_orb_calculator

        calculator = get_orb_calculator()
        logger.info(f"  Using: {C.CYAN}ORB v3 conservative{C.RESET}")
    except Exception as e:
        logger.error(f"{C.RED}Failed to initialize calculator: {e}{C.RESET}")
        db.close()
        sys.exit(1)

    supercell = tuple(args.supercell)
    logger.info(f"  Supercell: {supercell}")
    logger.info("")

    # Process entries
    logger.info(f"{C.BOLD}Running phonon calculations...{C.RESET}")
    logger.info(f"{C.DIM}{'-' * 50}{C.RESET}")

    num_stable = 0
    num_unstable = 0
    num_failed = 0

    for i, entry in enumerate(entries):
        e_hull = entry.e_above_hull or 0
        logger.info(
            f"{C.CYAN}[{i + 1}/{len(entries)}]{C.RESET} "
            f"{C.BOLD}{entry.formula:12s}{C.RESET}  "
            f"E_hull={e_hull * 1000:.1f} meV  "
            f"SG={entry.space_group_symbol}"
        )

        result = run_phonon_calculation(
            structure=entry,
            calculator=calculator,
            supercell=supercell,
        )

        if "error" in result:
            logger.info(f"  {C.YELLOW}⚠ Failed: {result['error']}{C.RESET}")
            num_failed += 1
            continue

        # Update database
        db._update_structure(
            structure_id=entry.id,
            is_dynamically_stable=result["is_stable"],
            num_imaginary_modes=result["num_imaginary_modes"],
            min_phonon_frequency=result["min_frequency"],
            max_phonon_frequency=result["max_frequency"],
            phonon_supercell=result["supercell"],
        )

        if result["is_stable"]:
            logger.info(f"  {C.GREEN}✓ Stable{C.RESET}")
            num_stable += 1
        else:
            logger.info(
                f"  {C.RED}✗ Unstable{C.RESET} "
                f"({result['num_imaginary_modes']} imaginary modes, "
                f"min freq: {result['min_frequency']:.2f} THz)"
            )
            num_unstable += 1

    # Summary
    logger.info("")
    logger.info(f"{C.BOLD}Summary{C.RESET}")
    logger.info(f"{C.DIM}{'-' * 50}{C.RESET}")
    logger.info(f"  Processed: {len(entries)}")
    logger.info(f"  Stable:    {C.GREEN}{num_stable}{C.RESET}")
    logger.info(f"  Unstable:  {C.RED}{num_unstable}{C.RESET}")
    if num_failed:
        logger.info(f"  Failed:    {C.YELLOW}{num_failed}{C.RESET}")

    db.close()
    logger.info("")
    logger.info(f"{C.GREEN}{C.BOLD}Done!{C.RESET}")


if __name__ == "__main__":
    main()
