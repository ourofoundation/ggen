#!/usr/bin/env python3
"""CLI for generating reports on chemical systems in the ggen database."""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen import SystemExplorer, Colors


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports on chemical systems in the ggen database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Co-Fe-Mn              # Report + phase diagram for Co-Fe-Mn
  %(prog)s --list                # List all systems in database
  %(prog)s Co-Fe-Mn --json       # Output as JSON
  %(prog)s Co-Fe-Mn --stable     # Show stable structures
  %(prog)s Co-Fe-Mn --exclude-p1 # Exclude P1 from phase diagram
  %(prog)s Co-Fe-Mn --tested-only    # Only phonon-tested structures
  %(prog)s Co-Fe-Mn --all-polymorphs # Include all polymorphs
        """,
    )

    parser.add_argument(
        "system",
        nargs="?",
        help="Chemical system to report on (e.g., Co-Fe-Mn)",
    )

    parser.add_argument(
        "--db",
        "-d",
        default="./ggen.db",
        help="Path to ggen database (default: ./ggen.db)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all chemical systems in the database",
    )

    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output report as JSON",
    )

    parser.add_argument(
        "--stable",
        action="store_true",
        help="Show stable structures (on hull)",
    )

    parser.add_argument(
        "--fully-stable",
        action="store_true",
        help="Show fully stable structures (on hull + phonon stable)",
    )

    parser.add_argument(
        "--tested-only",
        action="store_true",
        help="Only include phonon-tested structures in phase diagram",
    )

    parser.add_argument(
        "--all-polymorphs",
        action="store_true",
        help="Include all polymorphs, not just the lowest-energy per formula",
    )

    parser.add_argument(
        "--crystal-system",
        "-c",
        choices=[
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        ],
        help="Filter by crystal system",
    )

    parser.add_argument(
        "--space-group",
        "-s",
        help="Filter by space group (number or symbol)",
    )

    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Show stability breakdown",
    )

    parser.add_argument(
        "--energy-cutoff",
        type=float,
        default=0.150,
        metavar="EV",
        help="Energy above hull cutoff in eV/atom (default: 0.150)",
    )

    parser.add_argument(
        "--exclude-p1",
        action="store_true",
        help="Exclude P1 (triclinic) structures from phase diagram",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    # Handle color
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Check database exists
    db_path = Path(args.db)
    if not db_path.exists():
        print(
            f"{Colors.RED}Error:{Colors.RESET} Database not found: {db_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    explorer = SystemExplorer(str(db_path))

    try:
        # List systems mode
        if args.list:
            systems = explorer.list_systems()
            if not systems:
                print("No systems found in database")
                return

            print(f"{Colors.BOLD}Systems in database:{Colors.RESET}")
            for sys_name in systems:
                report = explorer.report(sys_name, energy_cutoff=args.energy_cutoff)
                within = report.stability.within_cutoff
                print(
                    f"{Colors.CYAN}{sys_name:<15}{Colors.RESET} "
                    f"{report.total_structures:>5} structures, "
                    f"{Colors.GREEN}{report.stability.on_hull:>3} on hull{Colors.RESET}, "
                    f"{Colors.YELLOW}{within:>3} near{Colors.RESET}"
                )
            return

        # Need a system for other operations
        if not args.system:
            parser.print_help()
            sys.exit(1)

        system = args.system

        # JSON report
        if args.json:
            report = explorer.report(system, energy_cutoff=args.energy_cutoff)
            print(json.dumps(report.to_dict(), indent=2))
            return

        # Stable structures list
        if args.stable:
            structures = explorer.get_stable_structures(system)
            print(
                f"{Colors.BOLD}On-hull structures in {system}:{Colors.RESET} ({len(structures)})\n"
            )
            for s in structures:
                dyn = ""
                if s.is_dynamically_stable is True:
                    dyn = f" {Colors.GREEN}✓ phonon{Colors.RESET}"
                elif s.is_dynamically_stable is False:
                    dyn = f" {Colors.RED}✗ phonon{Colors.RESET}"
                print(
                    f"  {Colors.CYAN}{s.formula:<15}{Colors.RESET} "
                    f"{Colors.MAGENTA}{s.space_group_symbol or '?':<10}{Colors.RESET} "
                    f"E={s.energy_per_atom:.4f}{dyn}"
                )
            return

        # Fully stable structures
        if args.fully_stable:
            structures = explorer.get_fully_stable_structures(system)
            print(
                f"{Colors.BOLD}Fully stable structures in {system}:{Colors.RESET} ({len(structures)})\n"
            )
            for s in structures:
                print(
                    f"  {Colors.GREEN}{s.formula:<15}{Colors.RESET} "
                    f"{Colors.MAGENTA}{s.space_group_symbol or '?':<10}{Colors.RESET} "
                    f"E={s.energy_per_atom:.4f}"
                )
            return

        # Crystal system filter
        if args.crystal_system:
            structures = explorer.get_structures_by_crystal_system(
                system, args.crystal_system
            )
            print(
                f"{Colors.BOLD}{args.crystal_system.title()} structures in {system}:{Colors.RESET} ({len(structures)})\n"
            )
            for s in structures[:20]:
                print(
                    f"  {Colors.CYAN}{s.formula:<15}{Colors.RESET} "
                    f"{Colors.MAGENTA}{s.space_group_symbol or '?':<10}{Colors.RESET} "
                    f"E={s.energy_per_atom:.4f}"
                )
            if len(structures) > 20:
                print(
                    f"  {Colors.DIM}... and {len(structures) - 20} more{Colors.RESET}"
                )
            return

        # Space group filter
        if args.space_group:
            try:
                sg = int(args.space_group)
            except ValueError:
                sg = args.space_group
            structures = explorer.get_structures_by_space_group(system, sg)
            print(
                f"{Colors.BOLD}Space group {args.space_group} structures in {system}:{Colors.RESET} ({len(structures)})\n"
            )
            for s in structures[:20]:
                print(
                    f"  {Colors.CYAN}{s.formula:<15}{Colors.RESET} "
                    f"E={s.energy_per_atom:.4f}"
                )
            if len(structures) > 20:
                print(
                    f"  {Colors.DIM}... and {len(structures) - 20} more{Colors.RESET}"
                )
            return

        # Stability breakdown
        if args.breakdown:
            breakdown = explorer.stability_breakdown(system)
            print(f"{Colors.BOLD}Stability breakdown for {system}:{Colors.RESET}\n")
            for category, structures in breakdown.items():
                color = {
                    "stable": Colors.GREEN,
                    "metastable": Colors.YELLOW,
                    "unstable": Colors.RED,
                    "untested": Colors.DIM,
                }.get(category, "")
                print(f"  {color}{category:<12}{Colors.RESET} {len(structures):>5}")
            return

        # Default: show report and generate phase diagram
        # Calculate filter counts before generating report
        exclude_sg = ["P1"] if args.exclude_p1 else None

        # Count structures that would be filtered
        all_structures = explorer.db.get_structures_for_subsystem(
            explorer.db.normalize_chemsys(system), valid_only=True
        )
        best_by_formula = explorer.db.get_best_structures_for_subsystem(
            explorer.db.normalize_chemsys(system)
        )

        # Determine which set of structures we're working with
        if args.all_polymorphs:
            structures_for_pd = all_structures
        else:
            structures_for_pd = list(best_by_formula.values())

        # Count P1 structures if --exclude-p1 is set
        p1_filtered = 0
        if args.exclude_p1:
            p1_filtered = sum(
                1 for s in structures_for_pd if s.space_group_symbol == "P1"
            )

        # Count additional polymorphs if --all-polymorphs is not set
        polymorphs_filtered = 0
        if not args.all_polymorphs:
            polymorphs_filtered = len(all_structures) - len(best_by_formula)

        # Generate report and set filter counts
        report = explorer.report(system, energy_cutoff=args.energy_cutoff)
        report.p1_filtered = p1_filtered
        report.polymorphs_filtered = polymorphs_filtered
        print(report.summary())

        # Build filename based on filters
        chemsys = explorer.db.normalize_chemsys(system)
        filename_parts = [chemsys, "phase_diagram"]

        if args.exclude_p1:
            filename_parts.append("noP1")
        if args.all_polymorphs:
            filename_parts.append("allpoly")
        if args.tested_only:
            filename_parts.append("tested")
        if args.energy_cutoff != 0.150:  # Only include if non-default
            # Format energy cutoff: 0.150 -> "cutoff150", 0.2 -> "cutoff200"
            cutoff_str = f"cutoff{int(args.energy_cutoff * 1000)}"
            filename_parts.append(cutoff_str)

        phase_diagram_filename = "_".join(filename_parts) + ".html"

        # Generate phase diagram
        saved_path, filter_counts = explorer.save_phase_diagram(
            system,
            output_path=phase_diagram_filename,
            show_unstable=args.energy_cutoff,
            exclude_space_groups=exclude_sg,
            tested_only=args.tested_only,
            all_polymorphs=args.all_polymorphs,
        )
        if saved_path:
            print()
            print(f"{Colors.GREEN}Phase diagram:{Colors.RESET} {saved_path}")
            png_path = saved_path.with_suffix(".png")
            if png_path.exists():
                print(f"{Colors.GREEN}Phase diagram:{Colors.RESET} {png_path}")

    finally:
        explorer.close()


if __name__ == "__main__":
    main()
