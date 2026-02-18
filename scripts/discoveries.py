#!/usr/bin/env python3
"""
Analyze new hull entries discovered by ggen vs Materials Project.

Compares the convex hull computed with all data (ggen + MP) against
what MP alone would produce, identifying structures where ggen found
a more stable phase than anything in the Materials Project.

Usage:
    python scripts/discoveries.py                    # All systems
    python scripts/discoveries.py Co-Fe-Mn           # Specific system
    python scripts/discoveries.py --json              # JSON output
    python scripts/discoveries.py --near-hull 0.050   # Include near-hull
    python scripts/discoveries.py -o results.json     # Write report file
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from ggen import Colors, StructureDatabase, StoredStructure


@dataclass
class Discovery:
    """A ggen structure that appears on the combined hull."""

    structure: StoredStructure
    e_above_hull: float
    category: str  # 'new_composition', 'beat_mp', 'elemental'
    mp_best_energy: Optional[float] = None
    energy_gain: Optional[float] = None  # eV/atom improvement over MP


@dataclass
class SystemDiscoveries:
    """Discovery analysis for one chemical system."""

    chemsys: str
    ggen_on_hull: List[Discovery] = field(default_factory=list)
    mp_on_hull: int = 0
    total_on_hull: int = 0
    ggen_near_hull: List[Discovery] = field(default_factory=list)
    mp_structures_count: int = 0
    ggen_structures_count: int = 0

    @property
    def new_compositions(self) -> List[Discovery]:
        return [d for d in self.ggen_on_hull if d.category == "new_composition"]

    @property
    def beat_mp(self) -> List[Discovery]:
        return [d for d in self.ggen_on_hull if d.category == "beat_mp"]

    @property
    def elementals(self) -> List[Discovery]:
        return [d for d in self.ggen_on_hull if d.category == "elemental"]

    @property
    def non_trivial_discoveries(self) -> List[Discovery]:
        return [d for d in self.ggen_on_hull if d.category != "elemental"]


@dataclass
class UniqueDiscovery:
    """A deduplicated discovery with the systems it appears in."""

    structure_id: str
    formula: str
    space_group: Optional[str]
    energy_per_atom: float
    category: str
    mp_best_energy: Optional[float]
    energy_gain: Optional[float]
    is_dynamically_stable: Optional[bool]
    is_p1: bool
    num_elements: int
    systems: List[str] = field(default_factory=list)


def analyze_system(
    db: StructureDatabase,
    chemsys: str,
    near_hull_cutoff: float = 0.0,
) -> SystemDiscoveries:
    """
    Analyze discoveries for a single chemical system.

    Computes the hull using all data, then categorizes each ggen hull
    entry based on whether MP had a competing structure at that formula.
    """
    chemsys = db.normalize_chemsys(chemsys)
    result = SystemDiscoveries(chemsys=chemsys)

    best_by_formula = db.get_best_structures_for_subsystem(chemsys)
    if not best_by_formula:
        return result

    mp_best = db.get_best_structures_for_subsystem(chemsys, source="mp")
    ggen_best = db.get_best_structures_for_subsystem(chemsys, source="ggen")
    result.mp_structures_count = len(mp_best)
    result.ggen_structures_count = len(ggen_best)

    entries = []
    entry_structure_pairs: List[Tuple[ComputedEntry, StoredStructure]] = []

    for structure in best_by_formula.values():
        if structure.energy_per_atom is None:
            continue
        comp = Composition(structure.formula)
        energy = structure.energy_per_atom * comp.num_atoms
        entry = ComputedEntry(comp, energy)
        entries.append(entry)
        entry_structure_pairs.append((entry, structure))

    if len(entries) < 2:
        return result

    try:
        pd = PhaseDiagram(entries)
    except Exception:
        return result

    for entry, structure in entry_structure_pairs:
        e_hull = pd.get_e_above_hull(entry)
        is_on_hull = e_hull < 1e-6
        is_near_hull = near_hull_cutoff > 0 and e_hull <= near_hull_cutoff

        if is_on_hull:
            result.total_on_hull += 1

        if structure.source != "ggen":
            if is_on_hull:
                result.mp_on_hull += 1
            continue

        formula_elements = set(Composition(structure.formula).get_el_amt_dict().keys())
        is_elemental = len(formula_elements) == 1

        mp_entry = mp_best.get(structure.formula)
        mp_energy = mp_entry.energy_per_atom if mp_entry else None

        if is_elemental:
            category = "elemental"
        elif mp_energy is None:
            category = "new_composition"
        else:
            category = "beat_mp"

        energy_gain = None
        if mp_energy is not None and structure.energy_per_atom is not None:
            energy_gain = mp_energy - structure.energy_per_atom

        disc = Discovery(
            structure=structure,
            e_above_hull=e_hull,
            category=category,
            mp_best_energy=mp_energy,
            energy_gain=energy_gain,
        )

        if is_on_hull:
            result.ggen_on_hull.append(disc)
        elif is_near_hull:
            result.ggen_near_hull.append(disc)

    result.ggen_on_hull.sort(key=lambda d: d.e_above_hull)
    result.ggen_near_hull.sort(key=lambda d: d.e_above_hull)

    return result


def deduplicate_discoveries(
    all_discoveries: List[SystemDiscoveries],
) -> List[UniqueDiscovery]:
    """
    Deduplicate discoveries across systems by structure_id.

    Each unique structure is attributed to all the systems it appears in,
    sorted by number of elements (most specific first).
    """
    seen: Dict[str, UniqueDiscovery] = {}

    for sys_disc in all_discoveries:
        for disc in sys_disc.non_trivial_discoveries:
            sid = disc.structure.id
            if sid not in seen:
                s = disc.structure
                formula_elements = set(
                    Composition(s.formula).get_el_amt_dict().keys()
                )
                seen[sid] = UniqueDiscovery(
                    structure_id=sid,
                    formula=s.formula,
                    space_group=s.space_group_symbol,
                    energy_per_atom=s.energy_per_atom,
                    category=disc.category,
                    mp_best_energy=disc.mp_best_energy,
                    energy_gain=disc.energy_gain,
                    is_dynamically_stable=s.is_dynamically_stable,
                    is_p1=s.space_group_symbol == "P1",
                    num_elements=len(formula_elements),
                )
            seen[sid].systems.append(sys_disc.chemsys)

    result = list(seen.values())
    result.sort(key=lambda d: (d.num_elements, d.formula))
    return result


def print_system_report(discoveries: SystemDiscoveries, near_hull_cutoff: float = 0.0):
    """Print a colored report for one chemical system."""
    C = Colors
    d = discoveries

    print(f"\n{C.BOLD}{C.CYAN}{d.chemsys}{C.RESET}")
    print(f"{C.DIM}{'─' * 60}{C.RESET}")
    print(
        f"  Structures: {C.CYAN}{d.ggen_structures_count} ggen{C.RESET}, "
        f"{C.MAGENTA}{d.mp_structures_count} mp{C.RESET} "
        f"{C.DIM}(best per formula){C.RESET}"
    )
    print(
        f"  On hull:    {C.GREEN}{len(d.ggen_on_hull)} ggen{C.RESET}, "
        f"{d.mp_on_hull} mp, "
        f"{d.total_on_hull} total"
    )

    non_trivial = d.non_trivial_discoveries
    if not non_trivial and not d.ggen_near_hull:
        print(f"  {C.DIM}No non-elemental ggen discoveries{C.RESET}")
        return

    if non_trivial:
        new_comp = d.new_compositions
        beat = d.beat_mp
        elementals = d.elementals

        parts = []
        if new_comp:
            parts.append(f"{C.GREEN}{len(new_comp)} new compositions{C.RESET}")
        if beat:
            parts.append(f"{C.YELLOW}{len(beat)} beat MP{C.RESET}")
        if elementals:
            parts.append(f"{C.DIM}{len(elementals)} elemental{C.RESET}")
        print(f"  Breakdown:  {', '.join(parts)}")
        print()

        for disc in non_trivial:
            s = disc.structure
            sg = s.space_group_symbol or "?"
            dyn = ""
            if s.is_dynamically_stable is True:
                dyn = f" {C.GREEN}phonon ✓{C.RESET}"
            elif s.is_dynamically_stable is False:
                dyn = f" {C.RED}phonon ✗{C.RESET}"

            p1_flag = f" {C.DIM}(P1){C.RESET}" if sg == "P1" else ""

            if disc.category == "new_composition":
                tag = f"{C.GREEN}NEW{C.RESET}"
            else:
                gain_str = ""
                if disc.energy_gain is not None:
                    gain_str = f" by {disc.energy_gain*1000:.1f} meV/atom"
                tag = f"{C.YELLOW}BEAT MP{gain_str}{C.RESET}"

            print(
                f"    {C.CYAN}{s.formula:<15}{C.RESET} "
                f"{C.MAGENTA}{sg:<10}{C.RESET} "
                f"E={s.energy_per_atom:.4f} eV/atom  "
                f"[{tag}]{dyn}{p1_flag}"
            )

    if near_hull_cutoff > 0 and d.ggen_near_hull:
        mev = int(near_hull_cutoff * 1000)
        print(f"\n  {C.BOLD}Near hull (within {mev} meV):{C.RESET}")
        for disc in d.ggen_near_hull:
            s = disc.structure
            sg = s.space_group_symbol or "?"
            ehull_mev = disc.e_above_hull * 1000
            print(
                f"    {C.CYAN}{s.formula:<15}{C.RESET} "
                f"{C.MAGENTA}{sg:<10}{C.RESET} "
                f"E_hull={ehull_mev:.1f} meV"
            )


def print_final_summary(
    unique: List[UniqueDiscovery],
    all_discoveries: List[SystemDiscoveries],
):
    """Print the deduplicated final summary section."""
    C = Colors

    if not unique:
        print(f"\n{C.DIM}No non-elemental discoveries found.{C.RESET}")
        return

    phonon_stable = [d for d in unique if d.is_dynamically_stable is True]
    phonon_unstable = [d for d in unique if d.is_dynamically_stable is False]
    phonon_untested = [d for d in unique if d.is_dynamically_stable is None]
    p1_entries = [d for d in unique if d.is_p1]
    non_p1 = [d for d in unique if not d.is_p1]

    new_comp = [d for d in unique if d.category == "new_composition"]
    beat_mp = [d for d in unique if d.category == "beat_mp"]

    print(f"\n{C.BOLD}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}Summary: Unique Discoveries{C.RESET}")
    print(f"{C.DIM}{'═' * 60}{C.RESET}")

    print(f"\n  {C.BOLD}{len(unique)} unique formulas{C.RESET} on hull across "
          f"{len(all_discoveries)} systems")
    print(f"    {C.GREEN}{len(new_comp)} new compositions{C.RESET} "
          f"(no MP entry at this formula)")
    if beat_mp:
        print(f"    {C.YELLOW}{len(beat_mp)} beat MP{C.RESET} "
              f"(lower energy than MP's best)")
    print()

    print(f"  {C.BOLD}Phonon status:{C.RESET}")
    print(f"    {C.GREEN}{len(phonon_stable)} confirmed stable{C.RESET}, "
          f"{C.RED}{len(phonon_unstable)} unstable{C.RESET}, "
          f"{C.DIM}{len(phonon_untested)} untested{C.RESET}")

    if p1_entries:
        print(f"\n  {C.BOLD}Symmetry:{C.RESET}")
        print(f"    {len(non_p1)} with higher symmetry, "
              f"{C.DIM}{len(p1_entries)} P1 (lower confidence){C.RESET}")

    # Print the unique discoveries, grouped by element count
    by_nelem: Dict[int, List[UniqueDiscovery]] = {}
    for d in unique:
        by_nelem.setdefault(d.num_elements, []).append(d)

    for nelem in sorted(by_nelem.keys()):
        entries = by_nelem[nelem]
        label = "Binary" if nelem == 2 else "Ternary" if nelem == 3 else f"{nelem}-ary"
        print(f"\n  {C.BOLD}{label} ({len(entries)}){C.RESET}")

        for d in entries:
            sg = d.space_group or "?"
            dyn = ""
            if d.is_dynamically_stable is True:
                dyn = f" {C.GREEN}✓{C.RESET}"
            elif d.is_dynamically_stable is False:
                dyn = f" {C.RED}✗{C.RESET}"

            p1_flag = f" {C.DIM}(P1){C.RESET}" if d.is_p1 else ""

            if d.category == "new_composition":
                tag = f"{C.GREEN}NEW{C.RESET}"
            else:
                gain_str = ""
                if d.energy_gain is not None:
                    gain_str = f" by {d.energy_gain*1000:.1f} meV"
                tag = f"{C.YELLOW}BEAT{gain_str}{C.RESET}"

            # Show the smallest system this discovery belongs to
            smallest_sys = min(d.systems, key=lambda s: len(s))

            print(
                f"    {C.CYAN}{d.formula:<15}{C.RESET} "
                f"{C.MAGENTA}{sg:<10}{C.RESET} "
                f"E={d.energy_per_atom:.4f}  "
                f"[{tag}]{dyn}{p1_flag}"
                f"  {C.DIM}({smallest_sys}){C.RESET}"
            )


def build_report(
    all_discoveries: List[SystemDiscoveries],
    unique: List[UniqueDiscovery],
    near_hull_cutoff: float = 0.0,
) -> Dict[str, Any]:
    """Build the full JSON-serializable report."""
    systems = {}

    for d in all_discoveries:
        entries = []
        for disc in d.ggen_on_hull:
            s = disc.structure
            entries.append({
                "formula": s.formula,
                "structure_id": s.id,
                "space_group": s.space_group_symbol,
                "energy_per_atom": s.energy_per_atom,
                "e_above_hull": disc.e_above_hull,
                "category": disc.category,
                "mp_best_energy": disc.mp_best_energy,
                "energy_gain_eV": disc.energy_gain,
                "is_dynamically_stable": s.is_dynamically_stable,
            })

        near_entries = []
        for disc in d.ggen_near_hull:
            s = disc.structure
            near_entries.append({
                "formula": s.formula,
                "structure_id": s.id,
                "space_group": s.space_group_symbol,
                "energy_per_atom": s.energy_per_atom,
                "e_above_hull": disc.e_above_hull,
                "category": disc.category,
                "is_dynamically_stable": s.is_dynamically_stable,
            })

        systems[d.chemsys] = {
            "ggen_structures": d.ggen_structures_count,
            "mp_structures": d.mp_structures_count,
            "ggen_on_hull": len(d.ggen_on_hull),
            "mp_on_hull": d.mp_on_hull,
            "total_on_hull": d.total_on_hull,
            "new_compositions": len(d.new_compositions),
            "beat_mp": len(d.beat_mp),
            "elemental": len(d.elementals),
            "entries": entries,
        }
        if near_entries:
            systems[d.chemsys]["near_hull_entries"] = near_entries

    unique_entries = []
    for d in unique:
        unique_entries.append({
            "formula": d.formula,
            "structure_id": d.structure_id,
            "space_group": d.space_group,
            "energy_per_atom": d.energy_per_atom,
            "category": d.category,
            "mp_best_energy": d.mp_best_energy,
            "energy_gain_eV": d.energy_gain,
            "is_dynamically_stable": d.is_dynamically_stable,
            "is_p1": d.is_p1,
            "num_elements": d.num_elements,
            "systems": d.systems,
        })

    phonon_stable = sum(1 for d in unique if d.is_dynamically_stable is True)
    phonon_unstable = sum(1 for d in unique if d.is_dynamically_stable is False)
    phonon_untested = sum(1 for d in unique if d.is_dynamically_stable is None)
    p1_count = sum(1 for d in unique if d.is_p1)

    return {
        "generated_at": datetime.now().isoformat(),
        "near_hull_cutoff_eV": near_hull_cutoff,
        "systems_analyzed": len(all_discoveries),
        "summary": {
            "unique_discoveries": len(unique),
            "new_compositions": sum(1 for d in unique if d.category == "new_composition"),
            "beat_mp": sum(1 for d in unique if d.category == "beat_mp"),
            "phonon_stable": phonon_stable,
            "phonon_unstable": phonon_unstable,
            "phonon_untested": phonon_untested,
            "p1_structures": p1_count,
            "higher_symmetry": len(unique) - p1_count,
        },
        "unique_discoveries": unique_entries,
        "per_system": systems,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze new hull entries discovered by ggen vs Materials Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # All systems
  %(prog)s Co-Fe-Mn                  # Specific system
  %(prog)s --json                    # JSON to stdout
  %(prog)s --near-hull 0.050         # Include within 50 meV of hull
  %(prog)s -o discoveries.json       # Write report file
        """,
    )

    parser.add_argument(
        "system",
        nargs="?",
        help="Chemical system to analyze (e.g., Co-Fe-Mn). Omit for all systems.",
    )

    parser.add_argument(
        "--db",
        "-d",
        default="./ggen.db",
        help="Path to ggen database (default: ./ggen.db)",
    )

    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output full report as JSON to stdout",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write JSON report to file",
    )

    parser.add_argument(
        "--near-hull",
        type=float,
        default=0.0,
        metavar="EV",
        help="Also show ggen structures within this cutoff of the hull (eV/atom)",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    db_path = Path(args.db)
    if not db_path.exists():
        print(
            f"{Colors.RED}Error:{Colors.RESET} Database not found: {db_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    db = StructureDatabase(str(db_path))
    C = Colors

    try:
        if args.system:
            systems = [db.normalize_chemsys(args.system)]
        else:
            systems = db.list_explored_systems()

        if not systems:
            print("No explored systems found in database.")
            sys.exit(0)

        # Analyze all systems
        all_discoveries = []
        for i, chemsys in enumerate(systems, 1):
            if not args.json and len(systems) > 1:
                print(
                    f"\r{C.DIM}Analyzing {chemsys}... ({i}/{len(systems)}){C.RESET}",
                    end="",
                    flush=True,
                )
            disc = analyze_system(db, chemsys, near_hull_cutoff=args.near_hull)
            all_discoveries.append(disc)

        if not args.json and len(systems) > 1:
            print("\r" + " " * 60 + "\r", end="")

        # Deduplicate across systems
        unique = deduplicate_discoveries(all_discoveries)

        # JSON mode: dump and exit
        if args.json:
            report = build_report(all_discoveries, unique, args.near_hull)
            print(json.dumps(report, indent=2))
            return

        # --- Header ---
        total_ggen_hull = sum(len(d.ggen_on_hull) for d in all_discoveries)
        total_mp_hull = sum(d.mp_on_hull for d in all_discoveries)

        print(f"\n{C.BOLD}Hull Discovery Analysis{C.RESET}")
        print(f"{C.DIM}{'═' * 60}{C.RESET}")
        print(f"  Systems analyzed:  {len(systems)}")
        print(f"  GGen on hull:      {C.GREEN}{total_ggen_hull}{C.RESET} "
              f"{C.DIM}(across all systems, with overlap){C.RESET}")
        print(f"  MP on hull:        {total_mp_hull}")

        if unique:
            print(f"\n  {C.BOLD}{C.GREEN}→ {len(unique)} unique non-elemental "
                  f"discoveries{C.RESET}")

        # --- Per-system reports ---
        for disc in all_discoveries:
            print_system_report(disc, near_hull_cutoff=args.near_hull)

        # --- Final deduplicated summary ---
        print_final_summary(unique, all_discoveries)

        # --- Write report file ---
        if args.output:
            report = build_report(all_discoveries, unique, args.near_hull)
            out_path = Path(args.output)
            out_path.write_text(json.dumps(report, indent=2))
            print(f"\n{C.GREEN}Report written to {out_path}{C.RESET}")
        else:
            default_name = "discoveries.json"
            report = build_report(all_discoveries, unique, args.near_hull)
            Path(default_name).write_text(json.dumps(report, indent=2))
            print(f"\n{C.DIM}Report written to {default_name}{C.RESET}")

        print()

    finally:
        db.close()


if __name__ == "__main__":
    main()
