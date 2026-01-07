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
  %(prog)s Co-Fe-Mn              # Report for Co-Fe-Mn system
  %(prog)s --list                # List all systems in database
  %(prog)s Co-Fe-Mn --json       # Output as JSON
  %(prog)s Co-Fe-Mn --stable     # Show stable structures
  %(prog)s Co-Fe-Mn --crystal-system cubic  # Filter by crystal system
        """,
    )
    
    parser.add_argument(
        "system",
        nargs="?",
        help="Chemical system to report on (e.g., Co-Fe-Mn)",
    )
    
    parser.add_argument(
        "--db", "-d",
        default="./ggen.db",
        help="Path to ggen database (default: ./ggen.db)",
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all chemical systems in the database",
    )
    
    parser.add_argument(
        "--json", "-j",
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
        "--untested",
        choices=["phonon", "hull"],
        help="Show structures that haven't been tested",
    )
    
    parser.add_argument(
        "--crystal-system", "-c",
        choices=["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"],
        help="Filter by crystal system",
    )
    
    parser.add_argument(
        "--space-group", "-s",
        help="Filter by space group (number or symbol)",
    )
    
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Show stability breakdown",
    )
    
    parser.add_argument(
        "--near-hull-cutoff",
        type=float,
        default=0.15,
        metavar="EV",
        help="Energy cutoff in eV/atom for 'near hull' (default: 0.15)",
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
        print(f"{Colors.RED}Error:{Colors.RESET} Database not found: {db_path}", file=sys.stderr)
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
                report = explorer.report(sys_name, near_hull_cutoff=args.near_hull_cutoff)
                print(f"  {Colors.CYAN}{sys_name:<15}{Colors.RESET} "
                      f"{report.total_structures:>5} structures, "
                      f"{Colors.GREEN}{report.stability.fully_stable:>3} stable{Colors.RESET}, "
                      f"{Colors.YELLOW}{report.stability.near_hull_stable:>3} near-hull{Colors.RESET}")
            return
        
        # Need a system for other operations
        if not args.system:
            parser.print_help()
            sys.exit(1)
        
        system = args.system
        
        # JSON report
        if args.json:
            report = explorer.report(system, near_hull_cutoff=args.near_hull_cutoff)
            print(json.dumps(report.to_dict(), indent=2))
            return
        
        # Stable structures list
        if args.stable:
            structures = explorer.get_stable_structures(system)
            print(f"{Colors.BOLD}On-hull structures in {system}:{Colors.RESET} ({len(structures)})\n")
            for s in structures:
                dyn = ""
                if s.is_dynamically_stable is True:
                    dyn = f" {Colors.GREEN}✓ phonon{Colors.RESET}"
                elif s.is_dynamically_stable is False:
                    dyn = f" {Colors.RED}✗ phonon{Colors.RESET}"
                print(f"  {Colors.CYAN}{s.formula:<15}{Colors.RESET} "
                      f"{Colors.MAGENTA}{s.space_group_symbol or '?':<10}{Colors.RESET} "
                      f"E={s.energy_per_atom:.4f}{dyn}")
            return
        
        # Fully stable structures
        if args.fully_stable:
            structures = explorer.get_fully_stable_structures(system)
            print(f"{Colors.BOLD}Fully stable structures in {system}:{Colors.RESET} ({len(structures)})\n")
            for s in structures:
                print(f"  {Colors.GREEN}{s.formula:<15}{Colors.RESET} "
                      f"{Colors.MAGENTA}{s.space_group_symbol or '?':<10}{Colors.RESET} "
                      f"E={s.energy_per_atom:.4f}")
            return
        
        # Untested structures
        if args.untested:
            structures = explorer.get_untested_structures(system, args.untested)
            label = "phonon-untested" if args.untested == "phonon" else "hull-untested"
            print(f"{Colors.BOLD}{label.title()} structures in {system}:{Colors.RESET} ({len(structures)})\n")
            for s in structures[:20]:  # Limit output
                print(f"  {Colors.YELLOW}{s.formula:<15}{Colors.RESET} "
                      f"{Colors.MAGENTA}{s.space_group_symbol or '?':<10}{Colors.RESET} "
                      f"E={s.energy_per_atom:.4f}")
            if len(structures) > 20:
                print(f"  {Colors.DIM}... and {len(structures) - 20} more{Colors.RESET}")
            return
        
        # Crystal system filter
        if args.crystal_system:
            structures = explorer.get_structures_by_crystal_system(system, args.crystal_system)
            print(f"{Colors.BOLD}{args.crystal_system.title()} structures in {system}:{Colors.RESET} ({len(structures)})\n")
            for s in structures[:20]:
                print(f"  {Colors.CYAN}{s.formula:<15}{Colors.RESET} "
                      f"{Colors.MAGENTA}{s.space_group_symbol or '?':<10}{Colors.RESET} "
                      f"E={s.energy_per_atom:.4f}")
            if len(structures) > 20:
                print(f"  {Colors.DIM}... and {len(structures) - 20} more{Colors.RESET}")
            return
        
        # Space group filter
        if args.space_group:
            try:
                sg = int(args.space_group)
            except ValueError:
                sg = args.space_group
            structures = explorer.get_structures_by_space_group(system, sg)
            print(f"{Colors.BOLD}Space group {args.space_group} structures in {system}:{Colors.RESET} ({len(structures)})\n")
            for s in structures[:20]:
                print(f"  {Colors.CYAN}{s.formula:<15}{Colors.RESET} "
                      f"E={s.energy_per_atom:.4f}")
            if len(structures) > 20:
                print(f"  {Colors.DIM}... and {len(structures) - 20} more{Colors.RESET}")
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
        
        # Default: show report
        report = explorer.report(system, near_hull_cutoff=args.near_hull_cutoff)
        print(report.summary())
        
    finally:
        explorer.close()


if __name__ == "__main__":
    main()

