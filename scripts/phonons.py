#!/usr/bin/env python3
"""
Backfill phonon (dynamical stability) calculations for existing database entries.

Focuses on low-energy entries near the convex hull that don't have phonon data yet.

Usage:
    # Backfill all systems, entries within 150 meV of hull
    python phonons.py

    # Specific chemical system
    python phonons.py --system Fe-Mn-Co

    # Only entries on the hull (0 meV)
    python phonons.py --e-above-hull 0.0

    # Re-run phonons for structures previously marked unstable
    # (useful after re-relaxing with relax.py)
    python phonons.py --rerun-unstable
    python phonons.py --system Fe-N --rerun-unstable

    # Limit number of structures to process
    python phonons.py --max-structures 10
"""

import argparse
import logging
import sys
import warnings
from typing import List, Optional, Tuple

# Type alias for supercell dimensions
SupercellDims = Tuple[int, int, int]

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
    e_above_hull_cutoff: float = 0.15,
    max_structures: Optional[int] = None,
    rerun_unstable: bool = False,
) -> List[StoredStructure]:
    """
    Get database entries that need phonon calculations.

    Args:
        db: Database connection
        chemical_system: Optional specific system (e.g., "Fe-Mn-Co"). If None, all systems.
        e_above_hull_cutoff: Maximum energy above hull in eV/atom
        max_structures: Optional limit on number of structures
        rerun_unstable: If True, also include structures previously marked as unstable

    Returns:
        List of StoredStructure objects needing phonon calculations
    """
    conn = db.conn

    # Build the stability filter
    if rerun_unstable:
        # Include both NULL (never tested) and unstable (0)
        stability_filter = (
            "(s.is_dynamically_stable IS NULL OR s.is_dynamically_stable = 0)"
        )
    else:
        # Only structures never tested
        stability_filter = "s.is_dynamically_stable IS NULL"

    if chemical_system:
        # Get entries for specific system
        chemsys = db.normalize_chemsys(chemical_system)
        query = f"""
            SELECT s.*, h.e_above_hull, h.is_on_hull
            FROM structures s
            JOIN hull_entries h ON s.id = h.structure_id
            WHERE h.chemsys = ?
              AND h.e_above_hull <= ?
              AND {stability_filter}
              AND s.cif_content IS NOT NULL
            ORDER BY h.e_above_hull ASC, s.energy_per_atom ASC
        """
        params: Tuple = (chemsys, e_above_hull_cutoff)
    else:
        # Get entries across all systems
        query = f"""
            SELECT DISTINCT s.*, MIN(h.e_above_hull) as e_above_hull, MAX(h.is_on_hull) as is_on_hull
            FROM structures s
            JOIN hull_entries h ON s.id = h.structure_id
            WHERE h.e_above_hull <= ?
              AND {stability_filter}
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


def get_adaptive_supercell(
    num_atoms: int,
    min_supercell_atoms: int = 150,
    max_dim: int = 5,
) -> Tuple[int, int, int]:
    """
    Determine supercell dimensions to reach a minimum atom count.
    Minimum supercell size is 3x3x3.

    Small supercells can produce spurious imaginary modes due to:
    - Force constant truncation at supercell boundaries
    - Incomplete Brillouin zone sampling
    - Unphysical periodic image interactions

    Args:
        num_atoms: Number of atoms in the unit cell
        min_supercell_atoms: Target minimum atoms in supercell (default: 150)
        max_dim: Maximum supercell dimension (default: 4)

    Returns:
        Tuple of supercell dimensions (n, n, n)
    """
    for n in range(3, max_dim + 1):
        if num_atoms * (n**3) >= min_supercell_atoms:
            return (n, n, n)
    return (max_dim, max_dim, max_dim)


def run_phonon_calculation(
    structure: StoredStructure,
    calculator,
    supercell: Optional[Tuple[int, int, int]] = None,
    min_supercell_atoms: int = 150,
) -> dict:
    """
    Run phonon calculation for a structure.

    Args:
        structure: Structure to calculate phonons for
        calculator: ASE calculator to use
        supercell: Explicit supercell dimensions, or None for adaptive sizing
        min_supercell_atoms: Minimum atoms in supercell when using adaptive sizing

    Returns dict with phonon results or error info.
    """
    pymatgen_structure = structure.get_structure()
    if pymatgen_structure is None:
        return {"error": "Could not load structure from CIF"}

    # Use adaptive supercell if not explicitly specified
    if supercell is None:
        supercell = get_adaptive_supercell(
            num_atoms=structure.num_atoms,
            min_supercell_atoms=min_supercell_atoms,
        )

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
        default=0.15,
        help="Max energy above hull in eV/atom (default: 0.15 = 150 meV)",
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
        default=None,
        metavar=("X", "Y", "Z"),
        help="Supercell dimensions (default: adaptive based on unit cell size)",
    )
    parser.add_argument(
        "--min-supercell-atoms",
        type=int,
        default=150,
        help="Minimum atoms in supercell for adaptive sizing (default: 150)",
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
    parser.add_argument(
        "--rerun-unstable",
        action="store_true",
        help="Re-run phonon calculations for structures previously marked unstable",
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
    filter_msg = f"E_hull ≤ {args.e_above_hull * 1000:.0f} meV"
    if args.rerun_unstable:
        filter_msg += " (including previously unstable)"
    else:
        filter_msg += " missing phonon data"
    logger.info(f"Finding entries with {filter_msg}...")

    entries = get_entries_needing_phonons(
        db=db,
        chemical_system=args.system,
        e_above_hull_cutoff=args.e_above_hull,
        max_structures=args.max_structures,
        rerun_unstable=args.rerun_unstable,
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

    supercell = tuple(args.supercell) if args.supercell else None
    if supercell:
        logger.info(f"  Supercell: {supercell} (fixed)")
    else:
        logger.info(f"  Supercell: adaptive (min {args.min_supercell_atoms} atoms)")
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
            min_supercell_atoms=args.min_supercell_atoms,
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

        sc = result["supercell"]
        sc_str = f"{sc[0]}×{sc[1]}×{sc[2]}"
        if result["is_stable"]:
            logger.info(f"  {C.GREEN}✓ Stable{C.RESET} (supercell: {sc_str})")
            num_stable += 1
        else:
            logger.info(
                f"  {C.RED}✗ Unstable{C.RESET} "
                f"({result['num_imaginary_modes']} imaginary modes, "
                f"min freq: {result['min_frequency']:.2f} THz, supercell: {sc_str})"
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
