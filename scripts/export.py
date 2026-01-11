#!/usr/bin/env python3
"""
Export top candidates from a chemical system as CIF files.

Exports structures ordered by energy above hull (lowest first),
with filters for dynamical stability and crystal system.

Usage:
    python scripts/export.py Co-Fe-Mn -n 10 -o ./export/
    python scripts/export.py Co-Fe-Mn -n 5 --crystal-systems tetragonal
    python scripts/export.py Co-Fe-Mn --max-ehull 0.05 --include-unstable
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen import Colors, StoredStructure, SystemExplorer
from ggen.report import get_crystal_system


def sanitize_filename(s: str) -> str:
    """Sanitize a string for use in filenames."""
    replacements = {
        "/": "-",
        "\\": "-",
        ":": "-",
        "*": "",
        "?": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "",
        " ": "_",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def format_ehull(e_above_hull: Optional[float]) -> str:
    """Format e_above_hull for filename (in meV)."""
    if e_above_hull is None:
        return "unknown"
    mev = e_above_hull * 1000
    if mev < 0.1:
        return "0meV"
    return f"{mev:.0f}meV"


def generate_cif_filename(
    structure: StoredStructure, include_ehull: bool = True
) -> str:
    """
    Generate a descriptive CIF filename.

    Format: formula_spacegroup_ehull.cif
    Example: Co3FeMn_Pm-3m_0meV.cif
    """
    parts = [structure.formula]

    # Add space group
    if structure.space_group_symbol:
        sg = sanitize_filename(structure.space_group_symbol)
        parts.append(sg)
    elif structure.space_group_number:
        parts.append(f"SG{structure.space_group_number}")

    # Add e_above_hull
    if include_ehull:
        parts.append(format_ehull(structure.e_above_hull))

    return "_".join(parts) + ".cif"


def export_candidates(
    chemical_system: str,
    output_dir: Path,
    n: int = 10,
    db_path: str = "./ggen.db",
    crystal_systems: Optional[List[str]] = None,
    dynamically_stable_only: bool = True,
    max_e_above_hull: Optional[float] = None,
    include_metadata: bool = True,
    verbose: bool = True,
) -> List[StoredStructure]:
    """
    Export top N candidates as CIF files.

    Args:
        chemical_system: Chemical system (e.g., "Co-Fe-Mn")
        output_dir: Directory to export CIF files to
        n: Number of candidates to export
        db_path: Path to the ggen database
        crystal_systems: Filter by crystal systems (e.g., ["tetragonal", "cubic"])
        dynamically_stable_only: Only export dynamically stable structures
        max_e_above_hull: Maximum energy above hull in eV/atom
        include_metadata: Write metadata JSON file
        verbose: Print progress information

    Returns:
        List of exported StoredStructure objects
    """
    C = Colors

    # Connect to database
    explorer = SystemExplorer(db_path)
    db = explorer.db
    chemsys = db.normalize_chemsys(chemical_system)

    if verbose:
        print(f"\n{C.BOLD}Exporting candidates from {C.CYAN}{chemsys}{C.RESET}")
        print(f"{C.DIM}{'─' * 50}{C.RESET}")

    # Get hull entries with a generous cutoff
    cutoff = max_e_above_hull if max_e_above_hull is not None else 1.0
    candidates = db.get_hull_entries(chemsys, e_above_hull_cutoff=cutoff)

    if verbose:
        print(f"  Candidates in database: {len(candidates)}")

    # Count untested structures (phonon stability not yet calculated)
    untested_count = sum(1 for s in candidates if s.is_dynamically_stable is None)

    # Filter by dynamical stability
    if dynamically_stable_only:
        before = len(candidates)
        candidates = [s for s in candidates if s.is_dynamically_stable is True]
        if verbose:
            print(f"  Dynamically stable: {len(candidates)}/{before}")
            if untested_count > 0:
                print(
                    f"  {C.YELLOW}Untested (phonons pending): {untested_count}{C.RESET}"
                )

    # Filter by crystal systems
    if crystal_systems:
        valid_systems = {
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        }
        crystal_systems_lower = [cs.lower() for cs in crystal_systems]
        invalid = [cs for cs in crystal_systems_lower if cs not in valid_systems]
        if invalid:
            print(
                f"{C.RED}Error: Invalid crystal system(s): {', '.join(invalid)}{C.RESET}"
            )
            print(f"Valid options: {', '.join(sorted(valid_systems))}")
            explorer.close()
            return []

        before = len(candidates)
        valid_systems_set = set(crystal_systems_lower)
        candidates = [
            s
            for s in candidates
            if s.space_group_number
            and get_crystal_system(s.space_group_number) in valid_systems_set
        ]
        if verbose:
            systems_str = ", ".join(cs.capitalize() for cs in crystal_systems_lower)
            print(f"  {systems_str} structures: {len(candidates)}/{before}")

    # Filter by max e_above_hull
    if max_e_above_hull is not None:
        before = len(candidates)
        candidates = [
            s
            for s in candidates
            if s.e_above_hull is not None and s.e_above_hull <= max_e_above_hull
        ]
        if verbose:
            print(
                f"  Within {max_e_above_hull*1000:.0f} meV of hull: {len(candidates)}/{before}"
            )

    # Sort by e_above_hull, then by energy_per_atom
    candidates.sort(
        key=lambda s: (
            s.e_above_hull if s.e_above_hull is not None else float("inf"),
            s.energy_per_atom if s.energy_per_atom is not None else float("inf"),
        )
    )

    # Take top N
    candidates = candidates[:n]

    if not candidates:
        if verbose:
            print(f"\n{C.YELLOW}No candidates match the criteria.{C.RESET}")
            if untested_count > 0:
                print(
                    f"\n{C.CYAN}Tip:{C.RESET} {untested_count} structure(s) have not been tested for dynamical stability."
                )
                print(f"     Run: {C.DIM}python phonons.py --system {chemsys}{C.RESET}")
        explorer.close()
        return []

    if verbose:
        print(f"\n  Exporting {C.GREEN}{len(candidates)}{C.RESET} candidates\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export CIF files
    exported = []
    metadata_list = []

    for i, structure in enumerate(candidates, 1):
        filename = generate_cif_filename(structure)
        filepath = output_dir / filename

        # Handle duplicate filenames
        counter = 1
        base_name = filepath.stem
        while filepath.exists():
            counter += 1
            filepath = output_dir / f"{base_name}_{counter}.cif"

        # Write CIF content
        if structure.cif_content:
            filepath.write_text(structure.cif_content)
            exported.append(structure)

            # Collect metadata
            crystal_sys = (
                get_crystal_system(structure.space_group_number)
                if structure.space_group_number
                else "unknown"
            )
            meta = {
                "filename": filepath.name,
                "formula": structure.formula,
                "space_group_number": structure.space_group_number,
                "space_group_symbol": structure.space_group_symbol,
                "crystal_system": crystal_sys,
                "energy_per_atom_eV": structure.energy_per_atom,
                "e_above_hull_eV": structure.e_above_hull,
                "e_above_hull_meV": (
                    structure.e_above_hull * 1000 if structure.e_above_hull else None
                ),
                "is_dynamically_stable": structure.is_dynamically_stable,
                "num_atoms": structure.num_atoms,
                "min_phonon_frequency_THz": structure.min_phonon_frequency,
                "structure_id": structure.id,
            }
            metadata_list.append(meta)

            if verbose:
                ehull_str = (
                    f"{structure.e_above_hull*1000:.1f} meV"
                    if structure.e_above_hull is not None
                    else "?"
                )
                stable_str = (
                    f"{C.GREEN}✓{C.RESET}"
                    if structure.is_dynamically_stable
                    else f"{C.RED}✗{C.RESET}"
                )
                print(f"  {i:2d}. {C.CYAN}{filepath.name}{C.RESET}")
                print(
                    f"      {structure.formula} | {crystal_sys} | E_hull: {ehull_str} | Stable: {stable_str}"
                )
        else:
            if verbose:
                print(
                    f"  {i:2d}. {C.YELLOW}Skipped{C.RESET} {structure.formula} (no CIF content)"
                )

    # Write metadata JSON
    if include_metadata and exported:
        metadata = {
            "chemical_system": chemsys,
            "export_date": datetime.now().isoformat(),
            "filters": {
                "dynamically_stable_only": dynamically_stable_only,
                "crystal_systems": crystal_systems,
                "max_e_above_hull_eV": max_e_above_hull,
                "top_n": n,
            },
            "num_exported": len(exported),
            "structures": metadata_list,
        }

        meta_path = output_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        if verbose:
            print(f"\n  {C.DIM}Metadata written to {meta_path.name}{C.RESET}")

    explorer.close()

    if verbose:
        print(
            f"\n{C.GREEN}✓ Exported {len(exported)} structures to {output_dir}{C.RESET}\n"
        )

    return exported


def generate_output_dirname(
    chemical_system: str,
    crystal_systems: Optional[List[str]] = None,
    max_e_above_hull: Optional[float] = None,
    dynamically_stable_only: bool = True,
) -> str:
    """Generate a descriptive output directory name based on filters."""
    parts = [chemical_system]

    if crystal_systems:
        parts.append("-".join(crystal_systems))

    if max_e_above_hull is not None:
        mev = int(max_e_above_hull * 1000)
        parts.append(f"{mev}meV")

    if not dynamically_stable_only:
        parts.append("incl-unstable")

    return "-".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Export top candidates from a chemical system as CIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Co-Fe-Mn -n 10
  %(prog)s Co-Fe-Mn -n 5 --crystal-systems tetragonal
  %(prog)s Co-Fe-Mn -n 5 --crystal-systems tetragonal cubic
  %(prog)s Co-Fe-Mn --max-ehull 0.05 --include-unstable

Output directories are auto-generated in exports/ based on filters:
  Co-Fe-Mn                          -> exports/Co-Fe-Mn/
  Co-Fe-Mn --crystal-systems tetragonal -> exports/Co-Fe-Mn-tetragonal/
  Co-Fe-Mn --crystal-systems tetragonal cubic -> exports/Co-Fe-Mn-tetragonal-cubic/
  Co-Fe-Mn --max-ehull 0.05         -> exports/Co-Fe-Mn-50meV/
        """,
    )

    parser.add_argument(
        "system",
        help="Chemical system to export (e.g., 'Co-Fe-Mn')",
    )

    parser.add_argument(
        "-n",
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to export (default: 10)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: auto-generated in exports/)",
    )

    parser.add_argument(
        "--db",
        "-d",
        default="./ggen.db",
        help="Path to ggen database (default: ./ggen.db)",
    )

    parser.add_argument(
        "--crystal-systems",
        type=str,
        nargs="+",
        choices=[
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        ],
        help="Filter by crystal system(s)",
    )

    parser.add_argument(
        "--max-ehull",
        type=float,
        help="Maximum energy above hull in eV/atom",
    )

    parser.add_argument(
        "--include-unstable",
        action="store_true",
        help="Include dynamically unstable structures",
    )

    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't write metadata.json file",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress output",
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

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        dirname = generate_output_dirname(
            chemical_system=args.system,
            crystal_systems=args.crystal_systems,
            max_e_above_hull=args.max_ehull,
            dynamically_stable_only=not args.include_unstable,
        )
        output_dir = Path("exports") / dirname

    try:
        exported = export_candidates(
            chemical_system=args.system,
            output_dir=output_dir,
            n=args.top,
            db_path=str(args.db),
            crystal_systems=args.crystal_systems,
            dynamically_stable_only=not args.include_unstable,
            max_e_above_hull=args.max_ehull,
            include_metadata=not args.no_metadata,
            verbose=not args.quiet,
        )
        sys.exit(0 if exported else 1)
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
