#!/usr/bin/env python3
"""
Explore a chemical system and generate a phase diagram.

Usage:
    python explore_system.py Fe-Mn-Si
    python explore_system.py Fe-Mn-Si --max-atoms 16 --num-trials 20

Run multiple systems in parallel:
    parallel python explore_system.py ::: Fe-Mn-Si Li-Co-O Fe-Sn-B
"""

import argparse
import logging
from pathlib import Path

from ggen import ChemistryExplorer


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
        default="./exploration_runs",
        help="Output directory for results (default: ./exploration_runs)",
    )
    parser.add_argument(
        "--load-previous",
        action="store_true",
        default=True,
        help="Load structures from previous runs (default: True)",
    )
    parser.add_argument(
        "--no-load-previous",
        action="store_false",
        dest="load_previous",
        help="Don't load structures from previous runs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip formulas that already exist (default: False)",
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
        "--e-above-hull",
        type=float,
        default=0.15,
        help="Energy above hull cutoff in eV/atom for stable phases (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress info logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Initialize explorer
    explorer = ChemistryExplorer(
        random_seed=args.seed,
        output_dir=args.output_dir,
    )

    logger.info(f"Starting exploration of {args.system}")

    # Run exploration
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
        load_previous_runs=args.load_previous,
        skip_existing_formulas=args.skip_existing,
        preserve_symmetry=args.preserve_symmetry,
    )

    # Log results
    print("\n" + "=" * 60)
    print(f"EXPLORATION RESULTS: {result.chemical_system}")
    print("=" * 60)
    print(f"Elements: {result.elements}")
    print(f"Total candidates attempted: {result.num_candidates}")
    print(f"Successful generations: {result.num_successful}")
    print(f"Failed generations: {result.num_failed}")
    print(f"Phases on convex hull: {len(result.hull_entries)}")
    print(f"Total time: {result.total_time_seconds:.1f}s")
    print(f"\nResults saved to: {result.run_directory}")
    print(f"Database: {result.database_path}")

    # Get stable candidates
    stable = explorer.get_stable_candidates(
        result, e_above_hull_cutoff=args.e_above_hull
    )

    print(
        f"\n{len(stable)} stable/near-stable phases (E_hull < {args.e_above_hull*1000:.0f} meV/atom):"
    )
    print("-" * 60)
    for c in stable:
        e_above = c.generation_metadata.get("e_above_hull", 0)
        source_run = c.generation_metadata.get("source_run", "current")
        print(
            f"  {c.formula:12s}  E={c.energy_per_atom:.4f} eV/atom  "
            f"SG={c.space_group_symbol:10s}  E_hull={e_above*1000:.1f} meV"
        )

    # Generate and save phase diagram
    if result.phase_diagram is not None:
        try:
            fig = explorer.plot_phase_diagram(result, show_unstable=args.e_above_hull)

            # Save the figure (plotly returns a plotly figure)
            pd_path_html = result.run_directory / "phase_diagram.html"
            fig.write_html(str(pd_path_html))
            print(f"\nPhase diagram saved to: {pd_path_html}")

            # Also try to save as PNG if kaleido is installed
            try:
                pd_path_png = result.run_directory / "phase_diagram.png"
                fig.write_image(str(pd_path_png), scale=2)
                print(f"Phase diagram PNG saved to: {pd_path_png}")
            except Exception:
                pass  # kaleido not installed, HTML is fine
        except Exception as e:
            logger.warning(f"Could not generate phase diagram: {e}")
    else:
        print("\nNo phase diagram available (need at least 2 valid candidates)")

    # Export summary
    summary = explorer.export_summary(
        result, output_path=result.run_directory / "summary.json"
    )
    print(f"Summary exported to: {result.run_directory / 'summary.json'}")

    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
