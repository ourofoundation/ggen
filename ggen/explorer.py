"""Chemistry exploration module for systematic structure generation and phase diagram analysis.

This module provides tools for exploring chemical spaces by generating candidate
structures across different stoichiometries and analyzing their thermodynamic stability.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.cif import CifWriter

from .ggen import GGen

logger = logging.getLogger(__name__)

# ===================== Data Classes =====================


@dataclass
class CandidateResult:
    """Result of a structure generation attempt."""

    formula: str
    stoichiometry: Dict[str, int]
    energy_per_atom: float
    total_energy: float
    num_atoms: int
    space_group_number: int
    space_group_symbol: str
    structure: Structure
    cif_path: Optional[Path] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class ExplorationResult:
    """Complete result of a chemistry exploration run."""

    chemical_system: str
    elements: List[str]
    num_candidates: int
    num_successful: int
    num_failed: int
    candidates: List[CandidateResult]
    phase_diagram: Optional[PhaseDiagram] = None
    hull_entries: List[CandidateResult] = field(default_factory=list)
    run_directory: Optional[Path] = None
    database_path: Optional[Path] = None
    total_time_seconds: float = 0.0


# ===================== ChemistryExplorer =====================


class ChemistryExplorer:
    """Systematic exploration of chemical spaces through structure generation.

    This class provides methods for:
    1. Parsing chemical systems (e.g., "Li-Co-O")
    2. Enumerating candidate stoichiometries
    3. Generating and optimizing structures for each composition
    4. Storing results in SQLite database and CIF files
    5. Building phase diagrams to identify stable candidates

    Example:
        >>> explorer = ChemistryExplorer()
        >>> result = explorer.explore("Li-Co-O", max_atoms=12, num_trials=5)
        >>> print(f"Found {len(result.hull_entries)} stable phases")
        >>> result.plot_phase_diagram()
    """

    def __init__(
        self,
        calculator=None,
        random_seed: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the chemistry explorer.

        Args:
            calculator: ASE calculator instance. If None, uses ORB calculator.
            random_seed: Optional random seed for reproducibility.
            output_dir: Base directory for storing results. If None, uses current directory.
        """
        self._calculator = calculator  # Store provided calculator (may be None)
        self._calculator_initialized = calculator is not None
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        # Will be set during exploration
        self._run_dir: Optional[Path] = None
        self._db_path: Optional[Path] = None
        self._db_conn: Optional[sqlite3.Connection] = None

    @property
    def calculator(self):
        """Lazily initialize and return the calculator.

        The MLIP model is only loaded once on first access, then reused
        for all subsequent structure generations.
        """
        if not self._calculator_initialized:
            from .calculator import get_orb_calculator

            logger.info("Initializing calculator (will be reused for all generations)")
            self._calculator = get_orb_calculator()
            self._calculator_initialized = True
        return self._calculator

    # -------------------- Chemical System Parsing --------------------

    def parse_chemical_system(self, system: str) -> List[str]:
        """Parse a chemical system string into a list of elements.

        Accepts formats like:
        - "Li-Co-O" (hyphen-separated)
        - "LiCoO" (concatenated element symbols)
        - ["Li", "Co", "O"] (list)

        Args:
            system: Chemical system specification.

        Returns:
            Sorted list of element symbols.

        Raises:
            ValueError: If the system cannot be parsed or contains invalid elements.
        """
        if isinstance(system, list):
            elements = system
        elif "-" in system:
            elements = [e.strip() for e in system.split("-")]
        else:
            # Parse concatenated symbols (e.g., "LiCoO")
            elements = []
            i = 0
            while i < len(system):
                if i + 1 < len(system) and system[i + 1].islower():
                    elements.append(system[i : i + 2])
                    i += 2
                else:
                    elements.append(system[i])
                    i += 1

        # Validate elements
        validated = []
        for el in elements:
            try:
                Element(el)
                validated.append(el)
            except ValueError:
                raise ValueError(f"Invalid element symbol: {el}")

        if len(validated) < 2:
            raise ValueError("Chemical system must contain at least 2 elements")

        return sorted(set(validated))

    # -------------------- Stoichiometry Enumeration --------------------

    def enumerate_stoichiometries(
        self,
        elements: List[str],
        max_atoms: int = 12,
        min_atoms: int = 2,
        max_per_element: Optional[int] = None,
        include_binaries: bool = True,
        include_ternaries: bool = True,
        require_all_elements: bool = False,
    ) -> List[Dict[str, int]]:
        """Enumerate candidate stoichiometries for a chemical system.

        Args:
            elements: List of element symbols.
            max_atoms: Maximum total atoms in the formula unit.
            min_atoms: Minimum total atoms in the formula unit.
            max_per_element: Maximum count per element (default: max_atoms).
            include_binaries: Whether to include binary compositions.
            include_ternaries: Whether to include ternary compositions (for 3+ element systems).
            require_all_elements: If True, only include compositions with all elements.

        Returns:
            List of stoichiometry dictionaries, e.g., [{"Li": 1, "Co": 1, "O": 2}, ...]
        """
        if max_per_element is None:
            max_per_element = max_atoms

        stoichiometries = []
        n_elements = len(elements)

        # Generate element subsets based on settings
        element_subsets = []

        if require_all_elements:
            element_subsets.append(tuple(elements))
        else:
            for r in range(2, n_elements + 1):
                if r == 2 and not include_binaries:
                    continue
                if r >= 3 and not include_ternaries:
                    continue
                for subset in combinations_with_replacement(elements, r):
                    # Convert to unique set
                    unique_subset = tuple(sorted(set(subset)))
                    if len(unique_subset) == r:
                        element_subsets.append(unique_subset)

            # Deduplicate
            element_subsets = list(set(element_subsets))

        # For each subset, enumerate stoichiometries
        for subset in element_subsets:
            subset_list = list(subset)
            n = len(subset_list)

            # Generate all combinations of counts
            def generate_counts(
                remaining_atoms: int, remaining_elements: int
            ) -> List[List[int]]:
                if remaining_elements == 1:
                    if 1 <= remaining_atoms <= max_per_element:
                        return [[remaining_atoms]]
                    return []

                results = []
                for count in range(
                    1,
                    min(remaining_atoms - remaining_elements + 1, max_per_element) + 1,
                ):
                    for rest in generate_counts(
                        remaining_atoms - count, remaining_elements - 1
                    ):
                        results.append([count] + rest)
                return results

            for total in range(max(min_atoms, n), max_atoms + 1):
                for counts in generate_counts(total, n):
                    stoich = {el: c for el, c in zip(subset_list, counts)}
                    stoichiometries.append(stoich)

        # Remove duplicates (same reduced formula)
        unique = {}
        for stoich in stoichiometries:
            comp = Composition(stoich)
            reduced = comp.reduced_formula
            if reduced not in unique:
                # Store the smallest representation using reduced_composition
                reduced_comp = comp.reduced_composition
                reduced_stoich = {
                    str(el): int(reduced_comp[el]) for el in reduced_comp.elements
                }
                unique[reduced] = reduced_stoich

        return list(unique.values())

    # -------------------- Database Management --------------------

    def _init_database(self, run_dir: Path) -> sqlite3.Connection:
        """Initialize SQLite database for storing exploration results."""
        db_path = run_dir / "exploration.db"
        conn = sqlite3.connect(str(db_path))

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                formula TEXT NOT NULL,
                stoichiometry TEXT NOT NULL,
                energy_per_atom REAL,
                total_energy REAL,
                num_atoms INTEGER,
                space_group_number INTEGER,
                space_group_symbol TEXT,
                cif_filename TEXT,
                is_valid INTEGER DEFAULT 1,
                error_message TEXT,
                generation_metadata TEXT,
                e_above_hull REAL,
                is_on_hull INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        conn.commit()
        self._db_path = db_path
        return conn

    def _to_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types and nested structures to JSON-serializable types."""
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(v) for v in obj]
        return obj

    def _save_candidate(
        self, conn: sqlite3.Connection, candidate: CandidateResult
    ) -> int:
        """Save a candidate to the database."""
        # Convert metadata to JSON-serializable types
        metadata = self._to_json_serializable(candidate.generation_metadata)

        cursor = conn.execute(
            """
            INSERT INTO candidates (
                formula, stoichiometry, energy_per_atom, total_energy,
                num_atoms, space_group_number, space_group_symbol,
                cif_filename, is_valid, error_message, generation_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.formula,
                json.dumps(candidate.stoichiometry),
                (
                    float(candidate.energy_per_atom)
                    if not np.isnan(candidate.energy_per_atom)
                    else None
                ),
                (
                    float(candidate.total_energy)
                    if not np.isnan(candidate.total_energy)
                    else None
                ),
                int(candidate.num_atoms),
                int(candidate.space_group_number),
                candidate.space_group_symbol,
                str(candidate.cif_path.name) if candidate.cif_path else None,
                1 if candidate.is_valid else 0,
                candidate.error_message,
                json.dumps(metadata),
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def _update_hull_status(
        self,
        conn: sqlite3.Connection,
        formula: str,
        e_above_hull: float,
        is_on_hull: bool,
    ) -> None:
        """Update the hull status for a candidate."""
        # Convert numpy types to native Python
        e_above_val = float(e_above_hull) if not np.isnan(e_above_hull) else None
        is_on_hull_val = 1 if bool(is_on_hull) else 0

        conn.execute(
            """
            UPDATE candidates
            SET e_above_hull = ?, is_on_hull = ?
            WHERE formula = ?
            """,
            (e_above_val, is_on_hull_val, formula),
        )
        conn.commit()

    # -------------------- Structure Generation --------------------

    def _generate_structure_for_stoichiometry(
        self,
        stoichiometry: Dict[str, int],
        num_trials: int = 10,
        optimize: bool = True,
        symmetry_bias: float = 0.0,
        crystal_systems: Optional[List[str]] = None,
        preserve_symmetry: bool = False,
    ) -> CandidateResult:
        """Generate and optimize a structure for a given stoichiometry.

        Args:
            stoichiometry: Element counts, e.g., {"Li": 1, "Co": 1, "O": 2}
            num_trials: Number of generation attempts per stoichiometry.
            optimize: Whether to optimize geometry.
            symmetry_bias: Controls preference for higher-symmetry crystal systems.
                0.0 = uniform distribution across orthorhombic and above.
                1.0 = strong preference for cubic/hexagonal. Default: 0.0
            crystal_systems: Optional list of crystal systems to restrict to.
            preserve_symmetry: If True, use symmetry-constrained relaxation to preserve
                high symmetry during optimization.

        Returns:
            CandidateResult with structure and energy information.
        """
        # Build formula string
        formula = "".join(
            f"{el}{count if count > 1 else ''}"
            for el, count in sorted(stoichiometry.items())
        )

        logger.info(f"Generating structure for {formula}")

        try:
            # Create a fresh GGen instance for each generation
            ggen = GGen(
                calculator=self.calculator,
                random_seed=self.random_seed,
                enable_trajectory=False,
            )

            # Generate crystal
            result = ggen.generate_crystal(
                formula=formula,
                num_trials=num_trials,
                optimize_geometry=optimize,
                multi_spacegroup=True,
                top_k_spacegroups=5,
                symmetry_bias=symmetry_bias,
                crystal_systems=crystal_systems,
                refine_symmetry=not preserve_symmetry,  # Don't need refinement if preserving
                preserve_symmetry=preserve_symmetry,
            )

            structure = ggen.get_structure()
            if structure is None:
                raise ValueError("No structure generated")

            energy = result["best_crystal_energy"]
            num_atoms = len(structure)
            energy_per_atom = energy / num_atoms

            return CandidateResult(
                formula=formula,
                stoichiometry=stoichiometry,
                energy_per_atom=energy_per_atom,
                total_energy=energy,
                num_atoms=num_atoms,
                space_group_number=result["final_space_group"],
                space_group_symbol=result["final_space_group_symbol"],
                structure=structure,
                generation_metadata={
                    "generation_stats": result.get("generation_stats", {}),
                    "score_breakdown": result.get("score_breakdown", {}),
                    "optimization_steps": result.get("optimization_steps", 0),
                },
            )

        except Exception as e:
            logger.warning(f"Failed to generate structure for {formula}: {e}")
            return CandidateResult(
                formula=formula,
                stoichiometry=stoichiometry,
                energy_per_atom=float("nan"),
                total_energy=float("nan"),
                num_atoms=0,
                space_group_number=0,
                space_group_symbol="",
                structure=None,  # type: ignore
                is_valid=False,
                error_message=str(e),
            )

    def _save_structure_cif(
        self, candidate: CandidateResult, structures_dir: Path
    ) -> Path:
        """Save structure to CIF file.

        Always saves the primitive cell to avoid supercell issues when
        loading CIFs back. Centered space groups (Im-3m, Fm-3m, etc.) have
        conventional cells 2-4x larger than primitive cells, which can cause
        issues with phase diagram calculations if the wrong cell is loaded.
        """
        filename = (
            f"{candidate.formula}_{candidate.space_group_symbol.replace('/', '-')}.cif"
        )
        cif_path = structures_dir / filename

        # Get primitive cell to avoid supercell issues on reload
        # This ensures consistent atom counts between save and load
        try:
            structure = candidate.structure.get_primitive_structure()
        except Exception:
            # Fall back to original structure if primitive conversion fails
            structure = candidate.structure

        try:
            cif_writer = CifWriter(structure, symprec=0.1)
        except Exception as e:
            # Fall back to saving without symmetry if spglib fails
            # (e.g., atoms too close together)
            logger.warning(
                f"Symmetry detection failed for {candidate.formula}, "
                f"saving without symmetry: {e}"
            )
            cif_writer = CifWriter(structure, symprec=None)

        with open(cif_path, "w") as f:
            f.write(str(cif_writer))

        return cif_path

    # -------------------- Phase Diagram Analysis --------------------

    def _generate_terminal_elements(
        self,
        elements: List[str],
        num_trials: int = 10,
        optimize: bool = True,
        symmetry_bias: float = 0.0,
        preserve_symmetry: bool = False,
    ) -> List[CandidateResult]:
        """Generate structures for pure elements (terminal entries for phase diagram).

        Note: Terminal elements are NOT constrained by crystal_systems parameter
        since pure elements need to find their natural lowest-energy structure
        (e.g., metals are often cubic, not hexagonal/tetragonal).

        Args:
            elements: List of element symbols.
            num_trials: Number of generation attempts per element.
            optimize: Whether to optimize geometry.
            symmetry_bias: Controls preference for higher-symmetry crystal systems.
            preserve_symmetry: If True, use symmetry-constrained relaxation.

        Returns:
            List of CandidateResult for each element.
        """
        terminal_candidates = []

        for element in elements:
            logger.info(f"Generating terminal element structure for {element}")

            # Try common stoichiometries for elemental structures
            # Start with single atom, then try 2 atoms (for dimers/pairs)
            best_candidate = None
            best_energy = float("inf")

            for num_atoms in [1, 2, 4]:
                stoich = {element: num_atoms}
                formula = f"{element}{num_atoms if num_atoms > 1 else ''}"

                try:
                    ggen = GGen(
                        calculator=self.calculator,
                        random_seed=self.random_seed,
                        enable_trajectory=False,
                    )

                    # Don't constrain crystal systems for terminal elements -
                    # they need to find their natural structure (e.g., cubic for metals)
                    result = ggen.generate_crystal(
                        formula=formula,
                        num_trials=num_trials,
                        optimize_geometry=optimize,
                        multi_spacegroup=True,
                        top_k_spacegroups=5,
                        symmetry_bias=symmetry_bias,
                        crystal_systems=None,  # Allow all crystal systems
                        refine_symmetry=not preserve_symmetry,
                        preserve_symmetry=preserve_symmetry,
                    )

                    structure = ggen.get_structure()
                    if structure is None:
                        continue

                    energy = result["best_crystal_energy"]
                    num_atoms_actual = len(structure)
                    energy_per_atom = energy / num_atoms_actual

                    candidate = CandidateResult(
                        formula=element,  # Use element symbol as formula for terminals
                        stoichiometry={element: 1},  # Normalized to 1 atom
                        energy_per_atom=energy_per_atom,
                        total_energy=energy,
                        num_atoms=num_atoms_actual,
                        space_group_number=result["final_space_group"],
                        space_group_symbol=result["final_space_group_symbol"],
                        structure=structure,
                        generation_metadata={
                            "is_terminal": True,
                            "original_formula": formula,
                        },
                    )

                    # Keep the lowest energy structure
                    if energy_per_atom < best_energy:
                        best_energy = energy_per_atom
                        best_candidate = candidate

                except Exception as e:
                    logger.warning(
                        f"Failed to generate {element} with {num_atoms} atoms: {e}"
                    )
                    continue

            if best_candidate is not None:
                terminal_candidates.append(best_candidate)
                logger.info(
                    f"Terminal {element}: E={best_candidate.energy_per_atom:.4f} eV/atom, "
                    f"SG={best_candidate.space_group_symbol}"
                )
            else:
                logger.error(f"Failed to generate any structure for element {element}")

        return terminal_candidates

    def _build_phase_diagram(
        self,
        candidates: List[CandidateResult],
        terminal_candidates: List[CandidateResult],
    ) -> Tuple[PhaseDiagram, List[CandidateResult]]:
        """Build a phase diagram from candidate structures.

        Args:
            candidates: List of successful candidate results (compounds).
            terminal_candidates: List of terminal element candidates.

        Returns:
            Tuple of (PhaseDiagram, list of hull entries from candidates)
        """
        valid_candidates = [
            c for c in candidates if c.is_valid and c.structure is not None
        ]

        if not valid_candidates:
            raise ValueError("No valid candidates for phase diagram")

        # Create computed entries for compounds
        entries = []
        candidate_map = {}  # Map entry_id to candidate

        for i, candidate in enumerate(valid_candidates):
            # Include space group in entry_id for better phase diagram labels
            sg_label = candidate.space_group_symbol.replace("/", "-")
            entry_id = f"{candidate.formula} ({sg_label})"

            # IMPORTANT: Use stored stoichiometry, not structure.composition
            # CIF loading can create supercells, causing composition mismatch
            # Also recalculate total_energy from energy_per_atom to ensure consistency
            composition = Composition(candidate.stoichiometry)
            num_atoms = sum(candidate.stoichiometry.values())
            total_energy = candidate.energy_per_atom * num_atoms

            entry = ComputedEntry(
                composition,
                total_energy,
                entry_id=entry_id,
            )
            entries.append(entry)
            candidate_map[entry_id] = candidate

        # Add terminal element entries (required for formation energy calculation)
        for terminal in terminal_candidates:
            if terminal.is_valid and terminal.structure is not None:
                sg_label = terminal.space_group_symbol.replace("/", "-")
                entry_id = f"{terminal.formula} ({sg_label})"

                # IMPORTANT: Use stored stoichiometry and recalculate total_energy
                # to avoid CIF supercell issues
                composition = Composition(terminal.stoichiometry)
                num_atoms = sum(terminal.stoichiometry.values())
                total_energy = terminal.energy_per_atom * num_atoms

                entry = ComputedEntry(
                    composition,
                    total_energy,
                    entry_id=entry_id,
                )
                entries.append(entry)
                # Don't add terminals to candidate_map - we only track compounds

        logger.info(
            f"Building phase diagram with {len(valid_candidates)} compounds + "
            f"{len(terminal_candidates)} terminal elements"
        )

        # Build phase diagram
        try:
            pd = PhaseDiagram(entries)
        except Exception as e:
            raise ValueError(f"Failed to build phase diagram: {e}")

        # Find hull entries (only for compound candidates, not terminals)
        hull_candidates = []
        for entry in entries:
            if entry.entry_id not in candidate_map:
                continue  # Skip terminal entries

            try:
                e_above = pd.get_e_above_hull(entry)
                candidate = candidate_map[entry.entry_id]

                # Update candidate with hull info
                candidate.generation_metadata["e_above_hull"] = e_above
                candidate.generation_metadata["is_on_hull"] = (
                    e_above < 0.001
                )  # ~1 meV tolerance

                if e_above < 0.001:
                    hull_candidates.append(candidate)

            except Exception as e:
                logger.warning(
                    f"Error computing hull distance for {entry.entry_id}: {e}"
                )

        return pd, hull_candidates

    # -------------------- Main Exploration Method --------------------

    def explore(
        self,
        chemical_system: str,
        max_atoms: int = 12,
        min_atoms: int = 2,
        num_trials: int = 10,
        optimize: bool = True,
        include_binaries: bool = True,
        include_ternaries: bool = True,
        require_all_elements: bool = False,
        max_stoichiometries: Optional[int] = None,
        run_name: Optional[str] = None,
        symmetry_bias: float = 0.0,
        crystal_systems: Optional[List[str]] = None,
        load_previous_runs: Union[bool, int] = False,
        skip_existing_formulas: bool = True,
        preserve_symmetry: bool = False,
    ) -> ExplorationResult:
        """Explore a chemical system by generating candidate structures.

        This is the main entry point for chemistry exploration. It:
        1. Parses the chemical system
        2. Optionally loads structures from previous runs
        3. Enumerates candidate stoichiometries
        4. Generates and optimizes structures for each (skipping already-explored ones)
        5. Stores all data in SQLite and CIF files
        6. Builds a phase diagram
        7. Returns stable candidates

        Args:
            chemical_system: Chemical system to explore (e.g., "Li-Co-O")
            max_atoms: Maximum atoms per formula unit.
            min_atoms: Minimum atoms per formula unit.
            num_trials: Generation attempts per stoichiometry.
            optimize: Whether to optimize geometries.
            include_binaries: Include binary compositions.
            include_ternaries: Include ternary compositions.
            require_all_elements: Only include compositions with all elements.
            max_stoichiometries: Limit number of stoichiometries to try.
            run_name: Optional name for the run directory.
            symmetry_bias: Controls preference for higher-symmetry crystal systems.
                0.0 = uniform distribution across orthorhombic and above.
                1.0 = strong preference for cubic/hexagonal. Default: 0.0
            crystal_systems: Optional list of crystal systems to restrict generation to.
                Valid values: "triclinic", "monoclinic", "orthorhombic", "tetragonal",
                "trigonal", "hexagonal", "cubic". If None, all systems are considered.
                Example: ["tetragonal"] to only generate tetragonal structures.
            load_previous_runs: Whether to load structures from previous runs of the
                same chemical system. Can be:
                - False (default): Don't load from previous runs
                - True: Load from all previous runs found
                - int: Load from the N most recent previous runs
            skip_existing_formulas: If True (default) and load_previous_runs is enabled,
                skip generating structures for formulas that were already successfully
                generated in previous runs. If False, regenerate all and keep the best.
            preserve_symmetry: If True, use symmetry-constrained relaxation to preserve
                high symmetry during optimization. This uses ASE's FixSymmetry constraint
                to keep atoms on their Wyckoff positions. Recommended when you want to
                maintain the generated space group. Default: False.

        Returns:
            ExplorationResult with all candidates, phase diagram, and stable phases.
        """
        import time

        start_time = time.time()

        # Parse chemical system
        elements = self.parse_chemical_system(chemical_system)
        chemsys = "-".join(elements)
        logger.info(f"Starting exploration of {chemsys}")

        # Setup run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"exploration_{chemsys}_{timestamp}"
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        structures_dir = run_dir / "structures"
        structures_dir.mkdir(exist_ok=True)

        self._run_dir = run_dir

        # Initialize database
        conn = self._init_database(run_dir)
        self._db_conn = conn

        # Save run metadata
        conn.execute(
            "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
            ("chemical_system", chemsys),
        )
        conn.execute(
            "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
            ("elements", json.dumps(elements)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
            ("started_at", datetime.now().isoformat()),
        )
        conn.commit()

        # Load structures from previous runs if requested
        previous_structures: Dict[str, CandidateResult] = {}
        if load_previous_runs:
            max_runs = None if load_previous_runs is True else int(load_previous_runs)
            previous_structures = self.load_structures_from_previous_runs(
                chemical_system=chemsys,
                max_runs=max_runs,
            )
            conn.execute(
                "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
                ("loaded_from_previous", json.dumps(list(previous_structures.keys()))),
            )
            conn.commit()

        # Enumerate stoichiometries
        stoichiometries = self.enumerate_stoichiometries(
            elements=elements,
            max_atoms=max_atoms,
            min_atoms=min_atoms,
            include_binaries=include_binaries,
            include_ternaries=include_ternaries,
            require_all_elements=require_all_elements,
        )

        if max_stoichiometries and len(stoichiometries) > max_stoichiometries:
            # Randomly sample stoichiometries
            indices = self.rng.choice(
                len(stoichiometries), size=max_stoichiometries, replace=False
            )
            stoichiometries = [stoichiometries[i] for i in indices]

        logger.info(f"Exploring {len(stoichiometries)} stoichiometries")

        # Generate structures for each stoichiometry
        candidates = []
        num_successful = 0
        num_failed = 0
        num_reused = 0

        for i, stoich in enumerate(stoichiometries):
            formula = "".join(
                f"{el}{count if count > 1 else ''}"
                for el, count in sorted(stoich.items())
            )

            # Check if we already have this formula from previous runs
            if formula in previous_structures and skip_existing_formulas:
                prev_candidate = previous_structures[formula]
                logger.info(
                    f"[{i+1}/{len(stoichiometries)}] Reusing {formula} from previous run "
                    f"(E={prev_candidate.energy_per_atom:.4f} eV/atom)"
                )

                # Copy the structure to the new run's structures directory
                if prev_candidate.structure is not None:
                    cif_path = self._save_structure_cif(prev_candidate, structures_dir)
                    prev_candidate.cif_path = cif_path

                # Mark as reused in metadata
                prev_candidate.generation_metadata["reused_from_previous"] = True

                # Save to database
                self._save_candidate(conn, prev_candidate)
                candidates.append(prev_candidate)
                num_successful += 1
                num_reused += 1
                continue

            logger.info(f"[{i+1}/{len(stoichiometries)}] Generating {formula}")

            candidate = self._generate_structure_for_stoichiometry(
                stoichiometry=stoich,
                num_trials=num_trials,
                optimize=optimize,
                symmetry_bias=symmetry_bias,
                crystal_systems=crystal_systems,
                preserve_symmetry=preserve_symmetry,
            )

            # If we have a previous structure and this one failed or is worse, use the previous
            if formula in previous_structures:
                prev_candidate = previous_structures[formula]
                if not candidate.is_valid or (
                    prev_candidate.is_valid
                    and prev_candidate.energy_per_atom < candidate.energy_per_atom
                ):
                    logger.info(
                        f"Using better structure from previous run for {formula} "
                        f"(prev={prev_candidate.energy_per_atom:.4f} vs "
                        f"new={candidate.energy_per_atom:.4f} eV/atom)"
                    )
                    candidate = prev_candidate
                    candidate.generation_metadata["reused_from_previous"] = True

            if candidate.is_valid and candidate.structure is not None:
                # Save CIF
                cif_path = self._save_structure_cif(candidate, structures_dir)
                candidate.cif_path = cif_path
                num_successful += 1
            else:
                num_failed += 1

            # Save to database
            self._save_candidate(conn, candidate)
            candidates.append(candidate)

        # Add remaining structures from previous runs that weren't in our enumeration
        if load_previous_runs:
            # Find formulas we already have from this run
            processed_formulas = {c.formula for c in candidates}
            # Elements will be handled as terminals, skip them
            element_formulas = set(elements)

            for formula, prev_candidate in previous_structures.items():
                if (
                    formula not in processed_formulas
                    and formula not in element_formulas
                ):
                    logger.info(
                        f"Including {formula} from previous run "
                        f"(E={prev_candidate.energy_per_atom:.4f} eV/atom)"
                    )
                    # Copy the structure to the new run's structures directory
                    if prev_candidate.structure is not None:
                        cif_path = self._save_structure_cif(
                            prev_candidate, structures_dir
                        )
                        prev_candidate.cif_path = cif_path

                    prev_candidate.generation_metadata["reused_from_previous"] = True
                    self._save_candidate(conn, prev_candidate)
                    candidates.append(prev_candidate)
                    num_successful += 1
                    num_reused += 1

            logger.info(
                f"Included {len(candidates)} total candidates "
                f"({num_reused} from previous runs)"
            )

        # Generate terminal element structures for phase diagram
        # Always generate fresh terminals and compare with previous runs to get the best
        terminal_candidates = []

        # Sanity bounds for terminal element energies (eV/atom)
        # Most metals are -4 to -10, some heavier metals up to -12
        # Anything outside this range is likely a calculation error
        MIN_REASONABLE_ENERGY = -12.0  # More negative than this is suspicious
        MAX_REASONABLE_ENERGY = -1.0  # Less negative than this is suspicious

        def _is_reasonable_terminal_energy(energy: float) -> bool:
            return MIN_REASONABLE_ENERGY <= energy <= MAX_REASONABLE_ENERGY

        logger.info(f"Generating terminal element structures for {elements}")
        new_terminals = self._generate_terminal_elements(
            elements=elements,
            num_trials=num_trials,
            optimize=optimize,
            symmetry_bias=symmetry_bias,
            preserve_symmetry=preserve_symmetry,
        )

        # Compare with previous runs and keep the better terminal
        # But reject terminals with obviously wrong energies
        for new_terminal in new_terminals:
            element = new_terminal.formula
            new_energy = new_terminal.energy_per_atom
            new_reasonable = _is_reasonable_terminal_energy(new_energy)

            if element in previous_structures:
                prev_terminal = previous_structures[element]
                prev_energy = prev_terminal.energy_per_atom
                prev_reasonable = _is_reasonable_terminal_energy(prev_energy)

                # Decision logic:
                # 1. If only one is reasonable, use that one
                # 2. If both reasonable, use lower energy
                # 3. If neither reasonable, use whichever is closer to reasonable range
                if prev_reasonable and not new_reasonable:
                    logger.warning(
                        f"New terminal {element} has unreasonable energy "
                        f"({new_energy:.4f} eV/atom), using previous "
                        f"({prev_energy:.4f} eV/atom)"
                    )
                    prev_terminal.generation_metadata["reused_from_previous"] = True
                    terminal_candidates.append(prev_terminal)
                elif new_reasonable and not prev_reasonable:
                    logger.warning(
                        f"Previous terminal {element} has unreasonable energy "
                        f"({prev_energy:.4f} eV/atom), using new "
                        f"({new_energy:.4f} eV/atom)"
                    )
                    terminal_candidates.append(new_terminal)
                elif prev_energy < new_energy:
                    logger.info(
                        f"Using terminal {element} from previous run "
                        f"(prev={prev_energy:.4f} vs new={new_energy:.4f} eV/atom)"
                    )
                    prev_terminal.generation_metadata["reused_from_previous"] = True
                    terminal_candidates.append(prev_terminal)
                else:
                    logger.info(
                        f"Using newly generated terminal {element} "
                        f"(new={new_energy:.4f} vs prev={prev_energy:.4f} eV/atom)"
                    )
                    terminal_candidates.append(new_terminal)
            else:
                if not new_reasonable:
                    logger.warning(
                        f"Terminal {element} has unreasonable energy: "
                        f"{new_energy:.4f} eV/atom (expected {MIN_REASONABLE_ENERGY} to "
                        f"{MAX_REASONABLE_ENERGY}). Phase diagram may be incorrect."
                    )
                else:
                    logger.info(
                        f"Terminal {element}: E={new_energy:.4f} eV/atom, "
                        f"SG={new_terminal.space_group_symbol}"
                    )
                terminal_candidates.append(new_terminal)

        # Save terminal structures
        for terminal in terminal_candidates:
            if terminal.is_valid and terminal.structure is not None:
                cif_path = self._save_structure_cif(terminal, structures_dir)
                terminal.cif_path = cif_path
                self._save_candidate(conn, terminal)

        # Build phase diagram
        phase_diagram = None
        hull_entries = []

        valid_candidates = [c for c in candidates if c.is_valid]
        valid_terminals = [t for t in terminal_candidates if t.is_valid]

        if len(valid_candidates) >= 1 and len(valid_terminals) == len(elements):
            try:
                phase_diagram, hull_entries = self._build_phase_diagram(
                    valid_candidates, valid_terminals
                )

                # Update database with hull info
                for candidate in valid_candidates:
                    e_above = candidate.generation_metadata.get(
                        "e_above_hull", float("nan")
                    )
                    is_on_hull = candidate.generation_metadata.get("is_on_hull", False)
                    self._update_hull_status(
                        conn, candidate.formula, e_above, is_on_hull
                    )

                logger.info(f"Found {len(hull_entries)} phases on the convex hull")

            except Exception as e:
                logger.warning(f"Failed to build phase diagram: {e}")
        else:
            missing_terminals = set(elements) - {t.formula for t in valid_terminals}
            if missing_terminals:
                logger.warning(
                    f"Cannot build phase diagram: missing terminal elements {missing_terminals}"
                )
            elif len(valid_candidates) < 1:
                logger.warning(
                    "Cannot build phase diagram: no valid compound candidates"
                )

        # Finalize
        total_time = time.time() - start_time

        conn.execute(
            "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
            ("completed_at", datetime.now().isoformat()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
            ("total_time_seconds", str(total_time)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
            ("num_reused_from_previous", str(num_reused)),
        )
        conn.commit()
        conn.close()

        reused_msg = f" ({num_reused} from previous runs)" if num_reused > 0 else ""
        logger.info(
            f"Exploration complete: {num_successful} successful{reused_msg}, "
            f"{len(hull_entries)} on hull, {total_time:.1f}s"
        )

        return ExplorationResult(
            chemical_system=chemsys,
            elements=elements,
            num_candidates=len(candidates),
            num_successful=num_successful,
            num_failed=num_failed,
            candidates=candidates,
            phase_diagram=phase_diagram,
            hull_entries=hull_entries,
            run_directory=run_dir,
            database_path=self._db_path,
            total_time_seconds=total_time,
        )

    # -------------------- Analysis & Visualization --------------------

    def get_stable_candidates(
        self, result: ExplorationResult, e_above_hull_cutoff: float = 0.025
    ) -> List[CandidateResult]:
        """Get candidates within a given energy above hull.

        Args:
            result: ExplorationResult from explore()
            e_above_hull_cutoff: Maximum energy above hull (eV/atom) to include.

        Returns:
            List of stable/near-stable candidates.
        """
        stable = []
        for candidate in result.candidates:
            if not candidate.is_valid:
                continue
            e_above = candidate.generation_metadata.get("e_above_hull", float("inf"))
            if e_above <= e_above_hull_cutoff:
                stable.append(candidate)

        # Sort by energy above hull
        stable.sort(
            key=lambda c: c.generation_metadata.get("e_above_hull", float("inf"))
        )
        return stable

    def plot_phase_diagram(
        self,
        result: ExplorationResult,
        show_unstable: float = 0.1,
        highlight_hull: bool = True,
    ):
        """Plot the phase diagram from exploration results.

        Args:
            result: ExplorationResult from explore()
            show_unstable: Energy cutoff for showing unstable phases (eV/atom)
            highlight_hull: Whether to highlight hull entries

        Returns:
            Plotly figure object (if phase diagram exists)
        """
        if result.phase_diagram is None:
            raise ValueError("No phase diagram available. Run explore() first.")

        try:
            fig = result.phase_diagram.get_plot(show_unstable=show_unstable)

            # Highlight hull entries if requested
            if highlight_hull and result.hull_entries:
                # This is handled by pymatgen's plotter
                pass

            return fig

        except Exception as e:
            logger.warning(f"Could not create phase diagram plot: {e}")
            raise

    def export_summary(
        self,
        result: ExplorationResult,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Export a summary of the exploration results.

        Args:
            result: ExplorationResult from explore()
            output_path: Optional path to save JSON summary.

        Returns:
            Summary dictionary.
        """
        summary = {
            "chemical_system": result.chemical_system,
            "elements": result.elements,
            "num_candidates": result.num_candidates,
            "num_successful": result.num_successful,
            "num_failed": result.num_failed,
            "num_hull_entries": len(result.hull_entries),
            "total_time_seconds": result.total_time_seconds,
            "run_directory": (
                str(result.run_directory) if result.run_directory else None
            ),
            "database_path": (
                str(result.database_path) if result.database_path else None
            ),
            "candidates": [],
            "hull_entries": [],
        }

        # Add candidate summaries
        for candidate in result.candidates:
            e_above = candidate.generation_metadata.get("e_above_hull")
            is_on_hull = candidate.generation_metadata.get("is_on_hull", False)
            cand_summary = {
                "formula": candidate.formula,
                "stoichiometry": self._to_json_serializable(candidate.stoichiometry),
                "energy_per_atom": self._to_json_serializable(
                    candidate.energy_per_atom
                ),
                "space_group": f"{candidate.space_group_symbol} (#{candidate.space_group_number})",
                "space_group_number": candidate.space_group_number,
                "space_group_symbol": candidate.space_group_symbol,
                "is_valid": bool(candidate.is_valid),
                "e_above_hull": (
                    self._to_json_serializable(e_above) if e_above is not None else None
                ),
                "is_on_hull": bool(is_on_hull),
                "cif_file": (
                    str(candidate.cif_path.name) if candidate.cif_path else None
                ),
                "reused_from_previous": candidate.generation_metadata.get(
                    "reused_from_previous", False
                ),
            }
            summary["candidates"].append(cand_summary)

        # Add hull entry summaries
        for candidate in result.hull_entries:
            summary["hull_entries"].append(
                {
                    "formula": candidate.formula,
                    "energy_per_atom": self._to_json_serializable(
                        candidate.energy_per_atom
                    ),
                    "space_group": f"{candidate.space_group_symbol} (#{candidate.space_group_number})",
                    "space_group_number": candidate.space_group_number,
                    "space_group_symbol": candidate.space_group_symbol,
                }
            )

        if output_path:
            output_path = Path(output_path)
            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2)

        return summary

    # -------------------- Load Previous Run --------------------

    def find_previous_runs(
        self, chemical_system: str, base_dir: Optional[Union[str, Path]] = None
    ) -> List[Path]:
        """Find previous exploration runs for the same chemical system.

        Args:
            chemical_system: Chemical system to search for (e.g., "Li-Co-O")
            base_dir: Directory to search in. Defaults to self.output_dir.

        Returns:
            List of run directories sorted by modification time (newest first).
        """
        elements = self.parse_chemical_system(chemical_system)
        chemsys = "-".join(elements)

        search_dir = Path(base_dir) if base_dir else self.output_dir

        if not search_dir.exists():
            return []

        # Find directories matching the pattern exploration_{chemsys}_*
        pattern = f"exploration_{chemsys}_*"
        runs = []

        for path in search_dir.glob(pattern):
            if path.is_dir() and (path / "exploration.db").exists():
                runs.append(path)

        # Sort by modification time (newest first)
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return runs

    def load_structures_from_previous_runs(
        self,
        chemical_system: str,
        max_runs: Optional[int] = None,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, CandidateResult]:
        """Load structures from previous exploration runs of the same chemical system.

        Args:
            chemical_system: Chemical system to search for (e.g., "Li-Co-O")
            max_runs: Maximum number of previous runs to load. None = all.
            base_dir: Directory to search in. Defaults to self.output_dir.

        Returns:
            Dictionary mapping formula -> CandidateResult with the best (lowest energy)
            structure for each formula found across all previous runs.
        """
        runs = self.find_previous_runs(chemical_system, base_dir)

        if max_runs is not None:
            runs = runs[:max_runs]

        logger.info(f"Found {len(runs)} previous runs for {chemical_system}")

        # Collect best structures by formula
        best_by_formula: Dict[str, CandidateResult] = {}

        for run_dir in runs:
            try:
                result = self.load_run(run_dir)
                logger.info(
                    f"Loaded {len(result.candidates)} candidates from {run_dir.name}"
                )

                for candidate in result.candidates:
                    if not candidate.is_valid or candidate.structure is None:
                        continue

                    formula = candidate.formula
                    if formula not in best_by_formula:
                        best_by_formula[formula] = candidate
                    elif (
                        candidate.energy_per_atom
                        < best_by_formula[formula].energy_per_atom
                    ):
                        # Keep the lower energy structure
                        best_by_formula[formula] = candidate

            except Exception as e:
                logger.warning(f"Failed to load run {run_dir}: {e}")
                continue

        logger.info(
            f"Loaded {len(best_by_formula)} unique structures from previous runs"
        )
        return best_by_formula

    @classmethod
    def load_run(cls, run_directory: Union[str, Path]) -> ExplorationResult:
        """Load a previous exploration run from its directory.

        Args:
            run_directory: Path to the run directory.

        Returns:
            ExplorationResult reconstructed from saved data.
        """
        run_dir = Path(run_directory)
        db_path = run_dir / "exploration.db"

        if not db_path.exists():
            raise ValueError(f"No database found at {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Load metadata
        metadata = {}
        for row in conn.execute("SELECT key, value FROM run_metadata"):
            metadata[row["key"]] = row["value"]

        chemical_system = metadata.get("chemical_system", "unknown")
        elements = json.loads(metadata.get("elements", "[]"))

        # Load candidates
        candidates = []
        structures_dir = run_dir / "structures"

        for row in conn.execute("SELECT * FROM candidates"):
            stoich = json.loads(row["stoichiometry"])
            gen_meta = (
                json.loads(row["generation_metadata"])
                if row["generation_metadata"]
                else {}
            )

            # Try to load structure from CIF
            # Use primitive=True to avoid conventional cell expansion
            # (centered space groups like Im-3m, Fm-3m have 2-4x larger conventional cells)
            structure = None
            cif_path = None
            if row["cif_filename"]:
                cif_path = structures_dir / row["cif_filename"]
                if cif_path.exists():
                    try:
                        structure = Structure.from_file(str(cif_path), primitive=True)
                    except Exception:
                        pass

            # Add hull info to metadata
            if row["e_above_hull"] is not None:
                gen_meta["e_above_hull"] = row["e_above_hull"]
            if row["is_on_hull"]:
                gen_meta["is_on_hull"] = bool(row["is_on_hull"])

            # Add source run info to metadata
            gen_meta["source_run"] = run_dir.name

            candidate = CandidateResult(
                formula=row["formula"],
                stoichiometry=stoich,
                energy_per_atom=row["energy_per_atom"] or float("nan"),
                total_energy=row["total_energy"] or float("nan"),
                num_atoms=row["num_atoms"] or 0,
                space_group_number=row["space_group_number"] or 0,
                space_group_symbol=row["space_group_symbol"] or "",
                structure=structure,
                cif_path=cif_path,
                generation_metadata=gen_meta,
                is_valid=bool(row["is_valid"]),
                error_message=row["error_message"],
            )
            candidates.append(candidate)

        conn.close()

        # Identify hull entries
        hull_entries = [
            c for c in candidates if c.generation_metadata.get("is_on_hull", False)
        ]

        return ExplorationResult(
            chemical_system=chemical_system,
            elements=elements,
            num_candidates=len(candidates),
            num_successful=sum(1 for c in candidates if c.is_valid),
            num_failed=sum(1 for c in candidates if not c.is_valid),
            candidates=candidates,
            phase_diagram=None,  # Would need to rebuild
            hull_entries=hull_entries,
            run_directory=run_dir,
            database_path=db_path,
            total_time_seconds=float(metadata.get("total_time_seconds", 0)),
        )
