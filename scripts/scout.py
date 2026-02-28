#!/usr/bin/env python3
"""
Scout: Systematically screen candidate elements for a chemical system template.

Runs shallow explorations across all candidate X elements in a group, then
ranks which systems are most promising based on convex hull results.

For long overnight runs, use jemalloc to prevent glibc malloc fragmentation
from ballooning memory:

    LD_PRELOAD=/usr/lib/libjemalloc.so python scout.py "Fe-Bi-{X}" --group 3d_metals

Usage:
    python scout.py "Fe-Bi-{X}" --group 3d_metals
    python scout.py "Fe-Bi-{X}" --group 3d_metals --crystal-systems tetragonal hexagonal
    python scout.py "Fe-Bi-{X}" --elements Co Mn Ni Cr V
    python scout.py "Fe-{X}-O" --group metalloids --min-fraction Fe:0.3
    python scout.py --list-groups
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

from ggen import Colors, StructureDatabase
from ggen.elements import get_element_group, list_groups, resolve_candidates
from ggen.scout import SystemScout

# Suppress noisy warnings from dependencies
warnings.filterwarnings(
    "ignore", category=UserWarning, module="pymatgen.core.structure"
)
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")
warnings.filterwarnings("ignore", category=UserWarning, module="orb_models.utils")
warnings.filterwarnings("ignore", category=UserWarning, module="ase.constraints")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy._lib._util")


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()
        return f"{record.levelname}: {record.getMessage()}"


def parse_fraction_constraints(constraints_list):
    """Parse fraction constraints from CLI format to dict."""
    if not constraints_list:
        return None
    result = {}
    for item in constraints_list:
        if ":" not in item:
            print(
                f"Error: Invalid fraction constraint format: '{item}'. "
                "Use 'Element:fraction' (e.g., 'Fe:0.4')",
                file=sys.stderr,
            )
            sys.exit(1)
        element, fraction_str = item.split(":", 1)
        try:
            fraction = float(fraction_str)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("Fraction must be between 0.0 and 1.0")
            result[element] = fraction
        except ValueError as e:
            print(
                f"Error: Invalid fraction value for {element}: {fraction_str}. {e}",
                file=sys.stderr,
            )
            sys.exit(1)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Scout: screen candidate elements for a chemical system template"
    )
    parser.add_argument(
        "template",
        type=str,
        nargs="?",
        help="Chemical system template with {X} placeholder (e.g., 'Fe-Bi-{X}')",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Named element group for candidates (e.g., '3d_metals', 'transition_metals')",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of candidate elements (e.g., Co Mn Ni Cr V)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help="Elements to exclude from candidates (in addition to template elements)",
    )
    parser.add_argument(
        "--crystal-systems",
        type=str,
        nargs="+",
        default=None,
        help="Target crystal systems for scoring (e.g., tetragonal hexagonal)",
    )
    parser.add_argument(
        "--min-fraction",
        type=str,
        nargs="+",
        default=None,
        help="Minimum element fraction constraints. Format: 'Fe:0.4' means Fe >= 40%%",
    )
    parser.add_argument(
        "--max-fraction",
        type=str,
        nargs="+",
        default=None,
        help="Maximum element fraction constraints. Format: 'Bi:0.2' means Bi <= 20%%",
    )
    parser.add_argument(
        "--shallow-trials",
        type=int,
        default=5,
        help="Generation attempts per stoichiometry (default: 5)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=12,
        help="Maximum atoms per unit cell (default: 12)",
    )
    parser.add_argument(
        "--min-atoms",
        type=int,
        default=2,
        help="Minimum atoms per unit cell (default: 2)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum optimization steps per structure (default: 400)",
    )
    parser.add_argument(
        "--relax-optimizer",
        type=str,
        choices=["fire", "lbfgs"],
        default="fire",
        help="Optimizer for relaxation (default: fire)",
    )
    parser.add_argument(
        "--e-above-hull",
        type=float,
        default=0.15,
        help="Energy above hull cutoff in eV/atom (default: 0.15)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./ggen.db",
        help="Path to unified structure database (default: ./ggen.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./runs",
        help="Output directory for run data (default: ./runs)",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=1,
        help="Parallel workers per exploration (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="Print available element groups and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which systems would be explored without running",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging from explorer",
    )

    args = parser.parse_args()
    C = Colors

    if args.no_color:
        Colors.disable()

    # --list-groups: print groups and exit
    if args.list_groups:
        groups = list_groups()
        print(f"\n{C.BOLD}Available element groups:{C.RESET}\n")
        for name, elements in sorted(groups.items()):
            print(f"  {C.CYAN}{name:<25}{C.RESET} {', '.join(elements)}")
        print()
        return

    # Validate required args
    if not args.template:
        parser.error("template is required (e.g., 'Fe-Bi-{X}')")

    if "{X}" not in args.template:
        parser.error("template must contain {X} placeholder (e.g., 'Fe-Bi-{X}')")

    if not args.group and not args.elements:
        parser.error("must provide --group or --elements")

    # Setup logging -- write to both console and a persistent file so output
    # survives tmux/OOM kills.
    log_level = logging.WARNING if args.quiet else logging.INFO
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())

    log_path = Path(args.output_dir) / "scout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    logging.basicConfig(
        level=log_level, handlers=[console_handler, file_handler], force=True
    )
    logging.getLogger(__name__).info("Scout log: %s", log_path)

    # Resolve candidate elements
    fixed_elements = SystemScout.parse_template(args.template)
    exclude = list(fixed_elements)
    if args.exclude:
        exclude.extend(args.exclude)

    groups = [args.group] if args.group else None
    candidates = resolve_candidates(
        groups=groups,
        elements=args.elements,
        exclude=exclude,
    )

    if not candidates:
        print(f"{C.RED}No candidate elements after applying exclusions.{C.RESET}")
        sys.exit(1)

    print(f"\n{C.BOLD}Scout Configuration:{C.RESET}")
    print(f"  Template: {args.template}")
    print(f"  Fixed elements: {', '.join(fixed_elements)}")
    if args.group:
        print(f"  Group: {args.group}")
    print(f"  Candidates ({len(candidates)}): {', '.join(candidates)}")
    if args.crystal_systems:
        print(f"  Target crystal systems: {', '.join(args.crystal_systems)}")

    # Dry run: show systems and exit
    if args.dry_run:
        print(f"\n{C.BOLD}Systems that would be explored:{C.RESET}")
        for el in candidates:
            chemsys = SystemScout.expand_template(args.template, el)
            print(f"  {chemsys}  (X={el})")
        print(f"\n  Total: {len(candidates)} systems")
        return

    # Parse fraction constraints
    min_fraction = parse_fraction_constraints(args.min_fraction)
    max_fraction = parse_fraction_constraints(args.max_fraction)

    # Connect to database
    db = StructureDatabase(args.db_path)

    try:
        scout = SystemScout(
            database=db,
            output_dir=args.output_dir,
            random_seed=args.seed,
        )

        scout.scan(
            template=args.template,
            candidates=candidates,
            num_trials=args.shallow_trials,
            max_atoms=args.max_atoms,
            min_atoms=args.min_atoms,
            crystal_systems=args.crystal_systems,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            e_above_hull_cutoff=args.e_above_hull,
            num_workers=args.workers,
            optimization_max_steps=args.max_steps,
            optimization_optimizer=args.relax_optimizer,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
