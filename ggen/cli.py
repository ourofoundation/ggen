"""
Command-line interface for GGen.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import GGen


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GGen: Crystal Generation and Mutation Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate crystal structures"
    )
    generate_parser.add_argument(
        "formula", help="Chemical formula (e.g., SiO2, BaTiO3)"
    )
    generate_parser.add_argument(
        "-s", "--space-group", type=int, help="Space group number"
    )
    generate_parser.add_argument(
        "-n", "--num-trials", type=int, default=10, help="Number of generation trials"
    )
    generate_parser.add_argument(
        "-o", "--optimize", action="store_true", help="Optimize geometry"
    )
    generate_parser.add_argument("-f", "--output-file", help="Output CIF file path")
    generate_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Mutate command
    mutate_parser = subparsers.add_parser("mutate", help="Apply mutations to structure")
    mutate_parser.add_argument("input_file", help="Input CIF file")
    mutate_parser.add_argument("-o", "--output-file", help="Output CIF file path")
    mutate_parser.add_argument("--scale", type=float, help="Scale lattice by factor")
    mutate_parser.add_argument("--jitter", type=float, help="Jitter sites with sigma")
    mutate_parser.add_argument(
        "--substitute",
        nargs=3,
        metavar=("FROM", "TO", "FRACTION"),
        help="Substitute element FROM TO with FRACTION",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze crystal structure")
    analyze_parser.add_argument("input_file", help="Input CIF file")
    analyze_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "generate":
            generate_crystal(args)
        elif args.command == "mutate":
            mutate_structure(args)
        elif args.command == "analyze":
            analyze_structure(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def generate_crystal(args) -> None:
    """Generate a crystal structure."""
    ggen = GGen()

    result = ggen.generate_crystal(
        formula=args.formula,
        space_group=args.space_group,
        num_trials=args.num_trials,
        optimize_geometry=args.optimize,
    )

    if args.json:
        # Remove binary data for JSON output
        json_result = {k: v for k, v in result.items() if k not in ["cif_base64"]}
        print(json.dumps(json_result, indent=2))
    else:
        output_file = args.output_file or f"{args.formula}.cif"
        with open(output_file, "w") as f:
            f.write(result["cif_content"])
        print(f"Generated crystal saved to {output_file}")
        print(
            f"Space group: {result['final_space_group_symbol']} #{result['final_space_group']}"
        )
        print(f"Energy: {result['best_crystal_energy']:.4f} eV")


def mutate_structure(args) -> None:
    """Apply mutations to a structure."""
    ggen = GGen()

    # Load structure from file
    ggen.load_structure_from_file(
        f"file://{Path(args.input_file).absolute()}", args.input_file
    )

    # Apply mutations
    if args.scale:
        ggen.scale_lattice(args.scale)
    if args.jitter:
        ggen.jitter_sites(sigma=args.jitter)
    if args.substitute:
        from_element, to_element, fraction = args.substitute
        ggen.substitute(from_element, to_element, float(fraction))

    # Get mutated structure
    structure = ggen.get_structure()
    if structure is None:
        raise ValueError("No structure loaded")

    # Save result
    output_file = args.output_file or f"mutated_{Path(args.input_file).stem}.cif"
    from pymatgen.io.cif import CifWriter

    CifWriter(structure).write_file(output_file)
    print(f"Mutated structure saved to {output_file}")


def analyze_structure(args) -> None:
    """Analyze a crystal structure."""
    ggen = GGen()

    # Load structure from file
    ggen.load_structure_from_file(
        f"file://{Path(args.input_file).absolute()}", args.input_file
    )

    # Get analysis
    summary = ggen.summary()

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Formula: {summary['formula']}")
        print(
            f"Space group: {summary['space_group']['symbol']} #{summary['space_group']['number']}"
        )
        print(
            f"Lattice: a={summary['lattice']['a']:.3f}, b={summary['lattice']['b']:.3f}, c={summary['lattice']['c']:.3f}"
        )
        print(
            f"Angles: α={summary['lattice']['alpha']:.1f}°, β={summary['lattice']['beta']:.1f}°, γ={summary['lattice']['gamma']:.1f}°"
        )
        print(f"Volume: {summary['lattice']['volume']:.3f} Å³")
        print(f"Density: {summary['lattice']['density']:.3f} g/cm³")
        if summary.get("energy") is not None:
            print(f"Energy: {summary['energy']:.4f} eV")


if __name__ == "__main__":
    main()
