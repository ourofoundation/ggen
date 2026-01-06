#!/usr/bin/env python3
"""
Explore a chemical system and generate a phase diagram.

Uses a unified database to share structures across chemical systems.
For example, Fe-Mn structures explored in Fe-Mn-Co will be reused when
exploring Fe-Mn-Sn.

Usage:
    python explore_system.py Fe-Mn-Si
    python explore_system.py Fe-Mn-Si --max-atoms 16 --num-trials 20

Run multiple systems in parallel:
    parallel python explore_system.py ::: Fe-Mn-Si Li-Co-O Fe-Sn-B
"""

import argparse
import logging
import sys
import warnings

from ggen import ChemistryExplorer, Colors, StructureDatabase

# Suppress pymatgen CIF parsing warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pymatgen.core.structure"
)
# Suppress orb_models torch dtype warnings
warnings.filterwarnings("ignore", category=UserWarning, module="orb_models.utils")


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    def format(self, record):
        # Only show the message for INFO level (cleaner output)
        if record.levelno == logging.INFO:
            return record.getMessage()
        # Show level for warnings/errors
        return f"{record.levelname}: {record.getMessage()}"


def get_system_stats(db: StructureDatabase, chemsys: str) -> dict:
    """Get statistics for a specific chemical system from the unified database."""
    chemsys_normalized = db.normalize_chemsys(chemsys)

    # Get all structures for this exact system (not subsystems)
    structures = db.get_structures_for_subsystem(chemsys_normalized)

    # Get hull entries for this system
    hull_entries = db.get_hull_entries(chemsys_normalized, e_above_hull_cutoff=0.0)
    near_hull = db.get_hull_entries(chemsys_normalized, e_above_hull_cutoff=0.150)

    # Count unique formulas
    formulas = set(s.formula for s in structures)

    return {
        "total_structures": len(structures),
        "unique_formulas": len(formulas),
        "on_hull": len(hull_entries),
        "near_hull": len(near_hull),
        "formulas": formulas,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Explore a chemical system and generate phase diagram"
    )
    parser.add_argument(
        "system",
        type=str,
        help="Chemical system to explore, e.g. 'Fe-Mn-Si' or 'Li-Co-O'",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=20,
        help="Maximum atoms per unit cell (default: 20)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=2,
        help="Minimum atoms per unit cell (default: 2)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=15,
        help="Number of trials per stoichiometry (default: 15)",
    )
    parser.add_argument(
        "--max-stoichiometries",
        type=int,
        default=100,
        help="Maximum stoichiometries to explore (default: 100)",
    )
    parser.add_argument(
        "--crystal-systems",
        type=str,
        nargs="+",
        default=None,
        help="Crystal systems to explore (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs",
        help="Output directory for CIF files and plots (default: ./runs)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./ggen.db",
        help="Path to unified structure database (default: ./ggen.db)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip formulas that already exist in database (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Regenerate all formulas even if they exist in database",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        default=False,
        help="Skip structure optimization (default: False)",
    )
    parser.add_argument(
        "--preserve-symmetry",
        action="store_true",
        default=True,
        help="Preserve symmetry during optimization (default: True)",
    )
    parser.add_argument(
        "--no-relax-all",
        action="store_true",
        default=False,
        help="Disable relaxing all trials (use legacy initial-energy selection)",
    )
    parser.add_argument(
        "--e-above-hull",
        type=float,
        default=0.15,
        help="Energy above hull cutoff in eV/atom for stable phases (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers for structure generation (default: 1)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--keep-in-memory",
        action="store_true",
        help="Keep structures in memory (default: save to CIF and free memory)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress info logging")

    args = parser.parse_args()

    # Disable colors if requested or not a tty
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    C = Colors

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger = logging.getLogger("ggen.explore")
    logger.handlers = [handler]
    logger.setLevel(log_level)
    logger.propagate = False

    # Suppress other loggers
    # logging.getLogger("ggen").setLevel(logging.WARNING)

    # Initialize unified database
    db = StructureDatabase(args.db_path)
    chemsys = db.normalize_chemsys(args.system)

    logger.info("")
    logger.info(f"{C.BOLD}{'=' * 60}{C.RESET}")
    logger.info(f"{C.BOLD}EXPLORING: {C.CYAN}{chemsys}{C.RESET}")
    logger.info(f"{C.BOLD}{'=' * 60}{C.RESET}")
    logger.info(f"{C.DIM}Database: {db.db_path}{C.RESET}")

    # Get stats BEFORE exploration
    global_stats = db.get_statistics()
    system_stats_before = get_system_stats(db, chemsys)

    logger.info("")
    logger.info(f"{C.BOLD}Database Overview{C.RESET}")
    logger.info(
        f"  Total structures (all systems): {C.WHITE}{global_stats['total_structures']}{C.RESET}"
    )
    logger.info(
        f"  Unique formulas (all systems):  {C.WHITE}{global_stats['unique_formulas']}{C.RESET}"
    )
    logger.info(
        f"  Chemical systems explored:      {C.WHITE}{global_stats['unique_chemsys']}{C.RESET}"
    )

    logger.info("")
    logger.info(f"{C.BOLD}{chemsys} Before This Run{C.RESET}")
    logger.info(
        f"  Structures in database: {C.WHITE}{system_stats_before['total_structures']}{C.RESET}"
    )
    logger.info(
        f"  Unique formulas:        {C.WHITE}{system_stats_before['unique_formulas']}{C.RESET}"
    )
    if system_stats_before["on_hull"] > 0:
        logger.info(
            f"  Phases on hull:         {C.WHITE}{system_stats_before['on_hull']}{C.RESET}"
        )

    # Initialize explorer with unified database
    explorer = ChemistryExplorer(
        random_seed=args.seed,
        output_dir=args.output_dir,
        database=db,
    )

    logger.info("")
    logger.info(f"{C.CYAN}Starting exploration...{C.RESET}")

    # Run exploration (uses unified database for cross-system structure sharing)
    result = explorer.explore(
        chemical_system=args.system,
        max_atoms=args.max_atoms,
        min_atoms=args.min_atoms,
        num_trials=args.num_trials,
        optimize=not args.no_optimize,
        include_binaries=True,
        include_ternaries=True,
        max_stoichiometries=args.max_stoichiometries,
        crystal_systems=args.crystal_systems,
        skip_existing_formulas=args.skip_existing,
        preserve_symmetry=args.preserve_symmetry,
        num_workers=args.workers,
        show_progress=not args.no_progress,
        keep_structures_in_memory=args.keep_in_memory,
        relax_all_trials=not args.no_relax_all,
        use_unified_database=True,
    )

    # Get stats AFTER exploration
    system_stats_after = get_system_stats(db, chemsys)
    global_stats_after = db.get_statistics()

    # Count new vs. reused from the result metadata
    new_generated = 0
    reused_from_db = 0
    for candidate in result.candidates:
        if candidate.is_valid:
            from_db = candidate.generation_metadata.get("loaded_from_unified_db", False)
            reused = candidate.generation_metadata.get("reused_from_previous", False)
            if from_db or reused:
                reused_from_db += 1
            else:
                new_generated += 1

    # Calculate database changes
    new_structures_in_db = (
        system_stats_after["total_structures"] - system_stats_before["total_structures"]
    )
    new_formulas_in_db = (
        system_stats_after["unique_formulas"] - system_stats_before["unique_formulas"]
    )

    # Print results
    logger.info("")
    logger.info(f"{C.BOLD}{'=' * 60}{C.RESET}")
    logger.info(f"{C.BOLD}RESULTS: {C.CYAN}{result.chemical_system}{C.RESET}")
    logger.info(f"{C.BOLD}{'=' * 60}{C.RESET}")

    logger.info("")
    logger.info(f"{C.BOLD}This Run{C.RESET}")
    logger.info(
        f"  Stoichiometries attempted: {C.WHITE}{result.num_candidates}{C.RESET}"
    )
    logger.info(f"  Newly generated:           {C.GREEN}{new_generated}{C.RESET}")
    logger.info(f"  Reused from database:      {C.BLUE}{reused_from_db}{C.RESET}")
    logger.info(f"  Failed generations:        {C.WHITE}{result.num_failed}{C.RESET}")
    logger.info(
        f"  Time elapsed:              {C.WHITE}{result.total_time_seconds:.1f}s{C.RESET}"
    )

    logger.info("")
    logger.info(f"{C.BOLD}Database Changes{C.RESET}")
    logger.info(f"  New structures added:   {C.GREEN}{new_structures_in_db}{C.RESET}")
    logger.info(f"  New formulas discovered:{C.GREEN} {new_formulas_in_db}{C.RESET}")

    logger.info("")
    logger.info(f"{C.BOLD}Phase Diagram ({chemsys}){C.RESET}")
    logger.info(
        f"  Total formulas on PD: {C.WHITE}{system_stats_after['unique_formulas']}{C.RESET}"
    )
    logger.info(
        f"  Phases on hull:       {C.GREEN}{system_stats_after['on_hull']}{C.RESET}"
    )
    logger.info(
        f"  On or near hull (≤150 meV):{C.YELLOW} {system_stats_after['near_hull']}{C.RESET}"
    )

    # Get ALL stable candidates from the database (not just this run)
    stable_from_db = db.get_hull_entries(chemsys, e_above_hull_cutoff=args.e_above_hull)

    # Track which formulas were newly generated in this run
    new_formulas = set()
    for candidate in result.candidates:
        if candidate.is_valid:
            from_db = candidate.generation_metadata.get("loaded_from_unified_db", False)
            reused = candidate.generation_metadata.get("reused_from_previous", False)
            if not from_db and not reused:
                new_formulas.add(candidate.formula)

    if stable_from_db:
        logger.info("")
        logger.info(
            f"{C.BOLD}Stable/Near-Stable Phases (E_hull < {args.e_above_hull * 1000:.0f} meV/atom){C.RESET}"
        )
        logger.info(f"{C.DIM}{'-' * 60}{C.RESET}")
        for s in stable_from_db:
            e_above = s.e_above_hull if s.e_above_hull is not None else 0
            # Color code: green for newly generated in this run, blue for from db
            if s.formula in new_formulas:
                formula_color = C.GREEN
                marker = "[new]"
            else:
                formula_color = C.BLUE
                marker = "[db]"
            logger.info(
                f"  {formula_color}{s.formula:12s}{C.RESET}  "
                f"E={s.energy_per_atom:.4f} eV/atom  "
                f"SG={s.space_group_symbol:10s}  "
                f"E_hull={e_above * 1000:.1f} meV  "
                f"{C.DIM}{marker}{C.RESET}"
            )

    # Generate and save phase diagram using ALL entries from the database
    # (not just what was found in this run)
    try:
        # Build the full phase diagram from all database entries for this system
        full_pd, e_above_hull_map = db.compute_hull(chemsys, update_database=True)

        if full_pd is not None:
            # Use pymatgen's plotter to create the figure
            fig = full_pd.get_plot(show_unstable=args.e_above_hull)

            # Save the figure
            pd_path_html = result.run_directory / "phase_diagram.html"
            fig.write_html(str(pd_path_html))
            logger.info("")
            logger.info(f"{C.BOLD}Output Files{C.RESET}")
            logger.info(f"  Phase diagram: {C.DIM}{pd_path_html}{C.RESET}")

            # Also try to save as PNG if kaleido is installed
            try:
                pd_path_png = result.run_directory / "phase_diagram.png"
                fig.write_image(str(pd_path_png), scale=2)
                logger.info(f"  Phase diagram: {C.DIM}{pd_path_png}{C.RESET}")
            except Exception:
                pass
        else:
            logger.info("")
            logger.warning(
                "No phase diagram available (need at least 2 valid candidates)"
            )
    except Exception as e:
        logger.warning(f"Could not generate phase diagram: {e}")

    # Export summary
    summary_path = result.run_directory / "summary.json"
    explorer.export_summary(result, output_path=summary_path)
    logger.info(f"  Summary:       {C.DIM}{summary_path}{C.RESET}")

    # Show global database stats
    logger.info("")
    logger.info(f"{C.BOLD}Database Totals{C.RESET}")
    logger.info(
        f"  All structures:       {C.WHITE}{global_stats_after['total_structures']}{C.RESET}"
    )
    logger.info(
        f"  All formulas:         {C.WHITE}{global_stats_after['unique_formulas']}{C.RESET}"
    )
    logger.info(
        f"  All chemical systems: {C.WHITE}{global_stats_after['unique_chemsys']}{C.RESET}"
    )

    logger.info("")
    logger.info(f"{C.BOLD}{'=' * 60}{C.RESET}")
    logger.info(f"{C.GREEN}{C.BOLD}EXPLORATION COMPLETE{C.RESET}")
    logger.info(f"{C.BOLD}{'=' * 60}{C.RESET}")
    logger.info("")

    # Close database connection
    db.close()

    return result


if __name__ == "__main__":
    main()
