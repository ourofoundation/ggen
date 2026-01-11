"""
System exploration and reporting module for GGen.

Provides tools for analyzing and reporting on crystal structures
stored in the database for specific chemical systems.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pymatgen.analysis.phase_diagram import PhaseDiagram

from .colors import Colors
from .database import StructureDatabase, StoredStructure


def get_crystal_system(sg_number: int) -> str:
    """Get crystal system name from space group number."""
    if sg_number is None:
        return "unknown"
    if 1 <= sg_number <= 2:
        return "triclinic"
    elif 3 <= sg_number <= 15:
        return "monoclinic"
    elif 16 <= sg_number <= 74:
        return "orthorhombic"
    elif 75 <= sg_number <= 142:
        return "tetragonal"
    elif 143 <= sg_number <= 167:
        return "trigonal"
    elif 168 <= sg_number <= 194:
        return "hexagonal"
    elif 195 <= sg_number <= 230:
        return "cubic"
    return "unknown"


@dataclass
class StabilityStats:
    """Statistics about structure stability."""

    total: int = 0

    # Thermodynamic stability (convex hull) - for best structure per formula
    on_hull: int = 0
    within_cutoff: int = 0  # within energy cutoff, excluding on_hull
    above_cutoff: int = 0  # beyond cutoff

    # Dynamical stability (phonons)
    dynamically_stable: int = 0
    dynamically_unstable: int = 0
    dynamically_untested: int = 0

    # Combined stability
    fully_stable: int = 0  # on_hull AND dynamically_stable

    @property
    def dynamically_tested(self) -> int:
        return self.dynamically_stable + self.dynamically_unstable


@dataclass
class SpaceGroupStats:
    """Statistics about space group distribution."""

    by_number: Dict[int, int] = field(default_factory=dict)
    by_symbol: Dict[str, int] = field(default_factory=dict)
    by_crystal_system: Dict[str, int] = field(default_factory=dict)

    # Top entries for quick access
    top_space_groups: List[Tuple[int, str, int]] = field(
        default_factory=list
    )  # (number, symbol, count)
    top_crystal_systems: List[Tuple[str, int]] = field(
        default_factory=list
    )  # (system, count)


@dataclass
class SystemReport:
    """Complete report for a chemical system."""

    chemical_system: str
    elements: List[str]

    total_structures: int = 0
    unique_formulas: int = 0

    stability: StabilityStats = field(default_factory=StabilityStats)
    space_groups: SpaceGroupStats = field(default_factory=SpaceGroupStats)

    # Energy statistics
    min_energy_per_atom: Optional[float] = None
    max_energy_per_atom: Optional[float] = None
    avg_energy_per_atom: Optional[float] = None

    # Energy cutoff used
    energy_cutoff: float = 0.150

    # Best structures
    hull_structures: List[StoredStructure] = field(default_factory=list)

    # Filter information (for display)
    p1_filtered: int = 0
    polymorphs_filtered: int = 0

    def summary(self) -> str:
        """Generate a text summary of the report."""
        C = Colors
        lines = []

        # Header
        lines.append(f"{C.BOLD}{C.CYAN}{self.chemical_system}{C.RESET}")
        lines.append(f"{C.DIM}{'─' * 50}{C.RESET}")
        lines.append(
            f"{self.total_structures} structures, {self.unique_formulas} unique formulas"
        )

        # Show filtering information right after counts
        if self.p1_filtered > 0 or self.polymorphs_filtered > 0:
            filter_lines = []
            if self.p1_filtered > 0:
                filter_lines.append(
                    f"  {C.DIM}{self.p1_filtered} filtered by --exclude-p1{C.RESET}"
                )
            if self.polymorphs_filtered > 0:
                filter_lines.append(
                    f"  {C.DIM}{self.polymorphs_filtered} additional polymorphs filtered "
                    f"(use --all-polymorphs to include){C.RESET}"
                )
            for line in filter_lines:
                lines.append(line)

        lines.append("")

        # Stability section
        cutoff_mev = int(self.energy_cutoff * 1000)
        lines.append(
            f"{C.BOLD}Thermodynamic Stability{C.RESET} {C.DIM}(per formula){C.RESET}"
        )
        lines.append(
            f"  {C.GREEN}{self.stability.on_hull} on hull{C.RESET}, "
            f"{C.YELLOW}{self.stability.within_cutoff} within {cutoff_mev} meV{C.RESET}, "
            f"{C.DIM}{self.stability.above_cutoff} above{C.RESET}"
        )
        lines.append("")

        # Dynamical
        lines.append(f"{C.BOLD}Dynamical Stability{C.RESET} {C.DIM}(phonon){C.RESET}")
        lines.append(
            f"  {C.GREEN}{self.stability.dynamically_stable} stable{C.RESET}, "
            f"{C.RED}{self.stability.dynamically_unstable} unstable{C.RESET}, "
            f"{C.DIM}{self.stability.dynamically_untested} untested{C.RESET}"
        )

        # Fully stable summary
        if self.stability.fully_stable > 0:
            lines.append(
                f"  {C.BOLD}{C.GREEN}→ {self.stability.fully_stable} fully stable{C.RESET} "
                f"{C.DIM}(on hull + phonon stable){C.RESET}"
            )
        lines.append("")

        # Space groups section
        lines.append(f"{C.BOLD}Crystal Systems{C.RESET}")
        for system, count in self.space_groups.top_crystal_systems:
            pct = (
                (count / self.total_structures * 100)
                if self.total_structures > 0
                else 0
            )
            bar_len = int(pct / 5)  # Scale to ~20 chars max
            bar = "█" * bar_len
            lines.append(
                f"  {system:<12} {C.CYAN}{bar:<10}{C.RESET} {count:>4} ({pct:>4.1f}%)"
            )
        lines.append("")

        # Top space groups
        lines.append(f"{C.BOLD}Top Space Groups{C.RESET}")
        for number, symbol, count in self.space_groups.top_space_groups[:5]:
            pct = (
                (count / self.total_structures * 100)
                if self.total_structures > 0
                else 0
            )
            lines.append(
                f"  {C.MAGENTA}{symbol:<10}{C.RESET} #{number:<3}  {count:>4} ({pct:>4.1f}%)"
            )
        lines.append("")

        # Energy
        if self.min_energy_per_atom is not None:
            lines.append(f"{C.BOLD}Energy{C.RESET} {C.DIM}(eV/atom){C.RESET}")
            lines.append(
                f"  min {C.GREEN}{self.min_energy_per_atom:>8.4f}{C.RESET}  "
                f"max {C.YELLOW}{self.max_energy_per_atom:>8.4f}{C.RESET}  "
                f"avg {self.avg_energy_per_atom:>8.4f}"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "chemical_system": self.chemical_system,
            "elements": self.elements,
            "total_structures": self.total_structures,
            "unique_formulas": self.unique_formulas,
            "energy_cutoff": self.energy_cutoff,
            "stability": {
                "total": self.stability.total,
                "on_hull": self.stability.on_hull,
                "within_cutoff": self.stability.within_cutoff,
                "above_cutoff": self.stability.above_cutoff,
                "dynamically_stable": self.stability.dynamically_stable,
                "dynamically_unstable": self.stability.dynamically_unstable,
                "dynamically_untested": self.stability.dynamically_untested,
                "fully_stable": self.stability.fully_stable,
            },
            "space_groups": {
                "by_crystal_system": dict(self.space_groups.by_crystal_system),
                "top_space_groups": [
                    {"number": n, "symbol": s, "count": c}
                    for n, s, c in self.space_groups.top_space_groups
                ],
            },
            "energy": {
                "min": self.min_energy_per_atom,
                "max": self.max_energy_per_atom,
                "avg": self.avg_energy_per_atom,
            },
        }


class SystemExplorer:
    """
    Explorer for analyzing and reporting on chemical systems in the database.

    Provides methods for generating reports, querying structures, and
    analyzing stability and space group distributions.

    Example:
        >>> explorer = SystemExplorer("./ggen.db")
        >>> report = explorer.report("Co-Fe-Mn")
        >>> print(report.summary())

        >>> # Get all stable structures
        >>> stable = explorer.get_stable_structures("Co-Fe-Mn")

        >>> # Filter by space group
        >>> cubic = explorer.get_structures_by_crystal_system("Co-Fe-Mn", "cubic")
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the system explorer.

        Args:
            db_path: Path to the ggen database. Defaults to ./ggen.db
        """
        self.db = StructureDatabase(db_path)

    def list_systems(self) -> List[str]:
        """List all chemical systems in the database."""
        return self.db.list_explored_systems()

    def report(self, chemical_system: str, energy_cutoff: float = 0.5) -> SystemReport:
        """
        Generate a comprehensive report for a chemical system.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")
            energy_cutoff: Energy cutoff in eV/atom for stability classification

        Returns:
            SystemReport with all statistics
        """
        chemsys = self.db.normalize_chemsys(chemical_system)
        elements = chemsys.split("-")

        # Get all structures for this system
        structures = self.db.get_structures_for_subsystem(chemsys, valid_only=True)

        if not structures:
            return SystemReport(
                chemical_system=chemsys,
                elements=elements,
                energy_cutoff=energy_cutoff,
            )

        # Get hull entries (on hull)
        hull_structures = self.db.get_hull_entries(chemsys, e_above_hull_cutoff=0.0)
        hull_ids = {s.id for s in hull_structures}

        # Get best structure per formula for hull stats
        best_by_formula = self.db.get_best_structures_for_subsystem(chemsys)

        # Calculate statistics
        stability = StabilityStats(total=len(structures))
        sg_by_number: Counter = Counter()
        sg_by_symbol: Counter = Counter()
        sg_by_system: Counter = Counter()

        energies = []
        unique_formulas = set()

        # Count hull stats from best structures per formula
        for formula, s in best_by_formula.items():
            is_on_hull = s.id in hull_ids
            if is_on_hull:
                stability.on_hull += 1
            elif s.e_above_hull is not None and s.e_above_hull <= energy_cutoff:
                stability.within_cutoff += 1
            else:
                stability.above_cutoff += 1

            # Fully stable = on hull AND dynamically stable
            if is_on_hull and s.is_dynamically_stable:
                stability.fully_stable += 1

        # Count all structures for other stats
        for s in structures:
            unique_formulas.add(s.formula)

            # Energy stats
            if s.energy_per_atom is not None:
                energies.append(s.energy_per_atom)

            # Dynamical stability
            if s.is_dynamically_stable is not None:
                if s.is_dynamically_stable:
                    stability.dynamically_stable += 1
                else:
                    stability.dynamically_unstable += 1
            else:
                stability.dynamically_untested += 1

            # Space group stats
            if s.space_group_number:
                sg_by_number[s.space_group_number] += 1
                crystal_sys = get_crystal_system(s.space_group_number)
                sg_by_system[crystal_sys] += 1

            if s.space_group_symbol:
                sg_by_symbol[s.space_group_symbol] += 1

        # Build space group stats
        space_groups = SpaceGroupStats(
            by_number=dict(sg_by_number),
            by_symbol=dict(sg_by_symbol),
            by_crystal_system=dict(sg_by_system),
        )

        # Top space groups (by number, with symbol)
        top_sg = sg_by_number.most_common(10)
        space_groups.top_space_groups = [
            (num, sg_by_symbol.most_common()[0][0] if sg_by_symbol else "?", count)
            for num, count in top_sg
        ]
        # Fix: get actual symbol for each space group
        sg_num_to_symbol = {}
        for s in structures:
            if s.space_group_number and s.space_group_symbol:
                sg_num_to_symbol[s.space_group_number] = s.space_group_symbol
        space_groups.top_space_groups = [
            (num, sg_num_to_symbol.get(num, "?"), count) for num, count in top_sg
        ]

        # Top crystal systems
        space_groups.top_crystal_systems = sg_by_system.most_common()

        # Energy stats
        min_e = min(energies) if energies else None
        max_e = max(energies) if energies else None
        avg_e = sum(energies) / len(energies) if energies else None

        return SystemReport(
            chemical_system=chemsys,
            elements=elements,
            total_structures=len(structures),
            unique_formulas=len(unique_formulas),
            stability=stability,
            space_groups=space_groups,
            min_energy_per_atom=min_e,
            max_energy_per_atom=max_e,
            avg_energy_per_atom=avg_e,
            hull_structures=hull_structures,
            energy_cutoff=energy_cutoff,
        )

    def get_stable_structures(
        self,
        chemical_system: str,
        include_near_hull: bool = False,
        e_above_hull_cutoff: float = 0.150,
    ) -> List[StoredStructure]:
        """
        Get thermodynamically stable structures (on or near hull).

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")
            include_near_hull: If True, include structures within cutoff of hull
            e_above_hull_cutoff: Energy cutoff in eV/atom for "near hull"

        Returns:
            List of stable StoredStructure objects
        """
        cutoff = e_above_hull_cutoff if include_near_hull else 0.0
        return self.db.get_hull_entries(chemical_system, e_above_hull_cutoff=cutoff)

    def get_dynamically_stable_structures(
        self,
        chemical_system: str,
    ) -> List[StoredStructure]:
        """
        Get structures that are dynamically stable (no imaginary phonon modes).

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")

        Returns:
            List of dynamically stable StoredStructure objects
        """
        structures = self.db.get_structures_for_subsystem(
            chemical_system, valid_only=True
        )
        return [s for s in structures if s.is_dynamically_stable is True]

    def get_fully_stable_structures(
        self,
        chemical_system: str,
    ) -> List[StoredStructure]:
        """
        Get structures that are both thermodynamically AND dynamically stable.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")

        Returns:
            List of fully stable StoredStructure objects
        """
        hull_entries = self.db.get_hull_entries(
            chemical_system, e_above_hull_cutoff=0.0
        )
        return [s for s in hull_entries if s.is_dynamically_stable is True]

    def get_untested_structures(
        self,
        chemical_system: str,
        test_type: str = "phonon",
    ) -> List[StoredStructure]:
        """
        Get structures that haven't been tested for stability.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")
            test_type: "phonon" for dynamical stability, "hull" for thermodynamic

        Returns:
            List of untested StoredStructure objects
        """
        structures = self.db.get_structures_for_subsystem(
            chemical_system, valid_only=True
        )

        if test_type == "phonon":
            return [s for s in structures if s.is_dynamically_stable is None]
        elif test_type == "hull":
            return [s for s in structures if s.e_above_hull is None]
        else:
            raise ValueError(f"Unknown test_type: {test_type}. Use 'phonon' or 'hull'")

    def get_structures_by_crystal_system(
        self,
        chemical_system: str,
        crystal_system: str,
    ) -> List[StoredStructure]:
        """
        Get structures belonging to a specific crystal system.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")
            crystal_system: One of: triclinic, monoclinic, orthorhombic,
                           tetragonal, trigonal, hexagonal, cubic

        Returns:
            List of StoredStructure objects in that crystal system
        """
        valid_systems = {
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        }
        crystal_system = crystal_system.lower()
        if crystal_system not in valid_systems:
            raise ValueError(
                f"Invalid crystal system: {crystal_system}. "
                f"Valid options: {sorted(valid_systems)}"
            )

        structures = self.db.get_structures_for_subsystem(
            chemical_system, valid_only=True
        )
        return [
            s
            for s in structures
            if s.space_group_number
            and get_crystal_system(s.space_group_number) == crystal_system
        ]

    def get_structures_by_space_group(
        self,
        chemical_system: str,
        space_group: Union[int, str],
    ) -> List[StoredStructure]:
        """
        Get structures with a specific space group.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")
            space_group: Space group number (1-230) or symbol (e.g., "P1", "Fm-3m")

        Returns:
            List of StoredStructure objects with that space group
        """
        structures = self.db.get_structures_for_subsystem(
            chemical_system, valid_only=True
        )

        if isinstance(space_group, int):
            return [s for s in structures if s.space_group_number == space_group]
        else:
            return [s for s in structures if s.space_group_symbol == space_group]

    def get_formulas(self, chemical_system: str) -> List[str]:
        """Get all unique formulas in a chemical system."""
        structures = self.db.get_structures_for_subsystem(
            chemical_system, valid_only=True
        )
        return sorted(set(s.formula for s in structures))

    def get_best_by_formula(
        self,
        chemical_system: str,
    ) -> Dict[str, StoredStructure]:
        """
        Get the best (lowest energy) structure for each formula.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")

        Returns:
            Dict mapping formula -> best StoredStructure
        """
        return self.db.get_best_structures_for_subsystem(chemical_system)

    def stability_breakdown(
        self, chemical_system: str
    ) -> Dict[str, List[StoredStructure]]:
        """
        Get structures grouped by stability category.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")

        Returns:
            Dict with keys: 'stable', 'metastable', 'unstable', 'untested'
            where 'stable' means on_hull AND dynamically_stable
        """
        structures = self.db.get_structures_for_subsystem(
            chemical_system, valid_only=True
        )
        hull_entries = self.db.get_hull_entries(
            chemical_system, e_above_hull_cutoff=0.0
        )
        hull_ids = {s.id for s in hull_entries}

        result = {
            "stable": [],  # on_hull AND dynamically_stable
            "metastable": [],  # on_hull but dynamically_unstable OR dynamically_stable but above_hull
            "unstable": [],  # above_hull AND dynamically_unstable
            "untested": [],  # missing either stability test
        }

        for s in structures:
            on_hull = s.id in hull_ids
            dyn_stable = s.is_dynamically_stable
            has_hull_data = on_hull or s.e_above_hull is not None

            if dyn_stable is None or not has_hull_data:
                result["untested"].append(s)
            elif on_hull and dyn_stable:
                result["stable"].append(s)
            elif on_hull or dyn_stable:
                result["metastable"].append(s)
            else:
                result["unstable"].append(s)

        return result

    def get_phase_diagram(
        self,
        chemical_system: str,
        update_database: bool = True,
        exclude_space_groups: Optional[List[str]] = None,
        tested_only: bool = False,
        all_polymorphs: bool = False,
    ) -> Tuple[Optional[PhaseDiagram], Dict[str, float], Dict[str, int]]:
        """
        Compute the phase diagram for a chemical system.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")
            update_database: If True, update hull entries in database
            exclude_space_groups: List of space group symbols to exclude (e.g., ["P1"])
            tested_only: If True, only include phonon-tested structures
            all_polymorphs: If True, include all structures not just best per formula

        Returns:
            Tuple of (PhaseDiagram, e_above_hull dict, filter_counts dict).
            PhaseDiagram is None if insufficient data.
        """
        return self.db.compute_hull(
            chemical_system,
            update_database=update_database,
            exclude_space_groups=exclude_space_groups,
            tested_only=tested_only,
            all_polymorphs=all_polymorphs,
        )

    def plot_phase_diagram(
        self,
        chemical_system: str,
        show_unstable: float = 0.1,
        update_database: bool = True,
        exclude_space_groups: Optional[List[str]] = None,
        tested_only: bool = False,
        all_polymorphs: bool = False,
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Generate a phase diagram plot for a chemical system.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")
            show_unstable: Energy cutoff for showing unstable phases (eV/atom)
            update_database: If True, update hull entries in database
            exclude_space_groups: List of space group symbols to exclude (e.g., ["P1"])
            tested_only: If True, only include phonon-tested structures
            all_polymorphs: If True, include all structures not just best per formula

        Returns:
            Tuple of (Plotly figure, filter_counts dict), or (None, {}) if insufficient data
        """
        pd, _, filter_counts = self.get_phase_diagram(
            chemical_system,
            update_database=update_database,
            exclude_space_groups=exclude_space_groups,
            tested_only=tested_only,
            all_polymorphs=all_polymorphs,
        )
        if pd is None:
            return None, filter_counts
        return pd.get_plot(show_unstable=show_unstable), filter_counts

    def save_phase_diagram(
        self,
        chemical_system: str,
        output_path: Optional[Union[str, Path]] = None,
        show_unstable: float = 0.1,
        save_png: bool = True,
        update_database: bool = True,
        exclude_space_groups: Optional[List[str]] = None,
        tested_only: bool = False,
        all_polymorphs: bool = False,
    ) -> Tuple[Optional[Path], Dict[str, int]]:
        """
        Generate and save a phase diagram for a chemical system.

        Args:
            chemical_system: Chemical system (e.g., "Co-Fe-Mn")
            output_path: Path to save HTML file. If None, uses
                         "{chemsys}_phase_diagram.html" in current directory.
            show_unstable: Energy cutoff for showing unstable phases (eV/atom)
            save_png: If True, also save PNG (requires kaleido)
            update_database: If True, update hull entries in database
            exclude_space_groups: List of space group symbols to exclude (e.g., ["P1"])
            tested_only: If True, only include phonon-tested structures
            all_polymorphs: If True, include all structures not just best per formula

        Returns:
            Tuple of (Path to saved HTML, filter_counts dict), or (None, {}) if insufficient data
        """
        fig, filter_counts = self.plot_phase_diagram(
            chemical_system,
            show_unstable=show_unstable,
            update_database=update_database,
            exclude_space_groups=exclude_space_groups,
            tested_only=tested_only,
            all_polymorphs=all_polymorphs,
        )
        if fig is None:
            return None, filter_counts

        chemsys = self.db.normalize_chemsys(chemical_system)

        if output_path is None:
            output_path = Path(f"{chemsys}_phase_diagram.html")
        else:
            output_path = Path(output_path)

        # Save HTML
        fig.write_html(str(output_path))

        # Try to save PNG if requested
        if save_png:
            try:
                png_path = output_path.with_suffix(".png")
                fig.write_image(str(png_path), scale=2)
            except Exception:
                pass  # kaleido not installed

        return output_path, filter_counts

    def close(self):
        """Close the database connection."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
