#!/usr/bin/env python3
"""
Run multiple chemical system explorations in parallel.

All explorations share a unified database, so structures from one system
(e.g., Fe-Mn from Fe-Mn-Co) are automatically reused in related systems
(e.g., Fe-Mn-Sn).

Usage:
    python explore_parallel.py Fe-Mn-Si Li-Co-O Fe-Sn-B
    python explore_parallel.py Fe-Mn-Si Li-Co-O --workers 2 --max-atoms 20

Or use GNU parallel directly:
    parallel python explore_system.py ::: Fe-Mn-Si Li-Co-O Fe-Sn-B
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_exploration(system: str, extra_args: list[str]) -> tuple[str, bool, str]:
    """Run exploration for a single system."""
    script_path = Path(__file__).parent / "explore_system.py"
    cmd = [sys.executable, str(script_path), system] + extra_args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return system, True, result.stdout
    except subprocess.CalledProcessError as e:
        return system, False, f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple chemical system explorations in parallel"
    )
    parser.add_argument(
        "systems", nargs="+", help="Chemical systems to explore, e.g. Fe-Mn-Si Li-Co-O"
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of systems)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=16,
        help="Maximum atoms per unit cell (default: 16)",
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
        "--inner-workers",
        type=int,
        default=1,
        help="Parallel workers per exploration (default: 1)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress individual exploration output"
    )

    args = parser.parse_args()

    # Build extra args to pass through
    extra_args = [
        "--max-atoms",
        str(args.max_atoms),
        "--num-trials",
        str(args.num_trials),
        "--max-stoichiometries",
        str(args.max_stoichiometries),
        "--output-dir",
        args.output_dir,
        "--db-path",
        args.db_path,
        "--workers",
        str(args.inner_workers),
    ]
    if args.quiet:
        extra_args.append("--quiet")
    if args.no_progress:
        extra_args.append("--no-progress")

    num_workers = args.workers or len(args.systems)

    print(f"Running {len(args.systems)} explorations with {num_workers} workers")
    print(f"Systems: {', '.join(args.systems)}")
    print(f"Unified database: {args.db_path}")
    print("=" * 60)

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_exploration, system, extra_args): system
            for system in args.systems
        }

        for future in as_completed(futures):
            system = futures[future]
            try:
                sys_name, success, output = future.result()
                results.append((sys_name, success))

                if not args.quiet:
                    print(f"\n{'=' * 60}")
                    print(f"COMPLETED: {sys_name} {'✓' if success else '✗'}")
                    print(output)
                else:
                    status = "✓" if success else "✗"
                    print(f"  {sys_name}: {status}")

            except Exception as e:
                results.append((system, False))
                print(f"  {system}: ✗ (Exception: {e})")

    # Summary
    print("\n" + "=" * 60)
    print("PARALLEL EXPLORATION SUMMARY")
    print("=" * 60)
    succeeded = sum(1 for _, s in results if s)
    failed = len(results) - succeeded
    print(f"Completed: {succeeded}/{len(results)}")
    if failed > 0:
        print(f"Failed: {', '.join(s for s, ok in results if not ok)}")
    print(f"Results in: {args.output_dir}")
    print(f"Unified database: {args.db_path}")


if __name__ == "__main__":
    main()
