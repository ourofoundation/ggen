"""Chemistry exploration module for systematic structure generation and phase diagram analysis.

This module provides tools for exploring chemical spaces by generating candidate
structures across different stoichiometries and analyzing their thermodynamic stability.
"""

from __future__ import annotations

import gc
import json
import logging
import signal
import sqlite3
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from tqdm import tqdm
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.io.cif import CifWriter

from .colors import Colors
from .ggen import GGen

# Import for type hints only to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import StructureDatabase

logger = logging.getLogger(__name__)

# ===================== Module-level helpers for parallel execution =====================


def _generate_structure_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for parallel structure generation.

    This is a module-level function to enable pickling for ProcessPoolExecutor.
    Returns a dictionary with results that can be converted to CandidateResult.
    """
    from .ggen import GGen
    from .calculator import get_orb_calculator

    stoichiometry = args["stoichiometry"]
    num_trials = args["num_trials"]
    optimize = args["optimize"]
    symmetry_bias = args["symmetry_bias"]
    crystal_systems = args["crystal_systems"]
    space_group = args.get("space_group")
    preserve_symmetry = args["preserve_symmetry"]
    random_seed = args["random_seed"]
    optimization_max_steps = args.get("optimization_max_steps", 400)

    # Build formula string
    formula = "".join(
        f"{el}{count if count > 1 else ''}"
        for el, count in sorted(stoichiometry.items())
    )

    try:
        # Each worker needs its own calculator instance
        calculator = get_orb_calculator()

        ggen = GGen(
            calculator=calculator,
            random_seed=random_seed,
            enable_trajectory=False,
        )

        result = ggen.generate_crystal(
            formula=formula,
            space_group=space_group,
            num_trials=num_trials,
            optimize_geometry=optimize,
            multi_spacegroup=space_group is None,  # Disable multi if specific SG given
            top_k_spacegroups=5,
            symmetry_bias=symmetry_bias,
            crystal_systems=crystal_systems,
            refine_symmetry=not preserve_symmetry,
            preserve_symmetry=preserve_symmetry,
            optimization_max_steps=optimization_max_steps,
        )

        structure = ggen.get_structure()
        if structure is None:
            raise ValueError("No structure generated")

        energy = result["best_crystal_energy"]
        num_atoms = len(structure)
        energy_per_atom = energy / num_atoms

        # Extract data before cleanup
        all_relaxed_trials = result.get("all_relaxed_trials", [])
        final_space_group = result["final_space_group"]
        final_space_group_symbol = result["final_space_group_symbol"]
        generation_stats = result.get("generation_stats", {})
        score_breakdown = result.get("score_breakdown", {})
        optimization_steps = result.get("optimization_steps", 0)

        # Clean up to free memory in this worker process
        ggen.cleanup()
        del result

        return {
            "formula": formula,
            "stoichiometry": stoichiometry,
            "energy_per_atom": energy_per_atom,
            "total_energy": energy,
            "num_atoms": num_atoms,
            "space_group_number": final_space_group,
            "space_group_symbol": final_space_group_symbol,
            "structure": structure,
            "generation_metadata": {
                "generation_stats": generation_stats,
                "score_breakdown": score_breakdown,
                "optimization_steps": optimization_steps,
            },
            "is_valid": True,
            "error_message": None,
            # Additional relaxed trials (polymorphs) for database storage
            "all_relaxed_trials": all_relaxed_trials,
        }

    except Exception as e:
        return {
            "formula": formula,
            "stoichiometry": stoichiometry,
            "energy_per_atom": float("nan"),
            "total_energy": float("nan"),
            "num_atoms": 0,
            "space_group_number": 0,
            "space_group_symbol": "",
            "structure": None,
            "generation_metadata": {},
            "is_valid": False,
            "error_message": str(e),
        }


# ===================== Data Classes =====================


@dataclass
class CandidateResult:
    """Result of a structure generation attempt.

    Note: For memory efficiency, the `structure` field may be None even for valid
    candidates. Use `get_structure()` to lazily load from CIF when needed.
    """

    formula: str
    stoichiometry: Dict[str, int]
    energy_per_atom: float
    total_energy: float
    num_atoms: int
    space_group_number: int
    space_group_symbol: str
    structure: Optional[Structure] = None
    cif_path: Optional[Path] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    error_message: Optional[str] = None

    # Dynamical stability (phonon) properties
    is_dynamically_stable: Optional[bool] = None
    phonon_result: Optional[Any] = None  # PhononResult from phonons module

    def get_structure(self) -> Optional[Structure]:
        """Get the structure, loading from CIF if not in memory."""
        if self.structure is not None:
            return self.structure
        if self.cif_path is not None and self.cif_path.exists():
            try:
                self.structure = Structure.from_file(str(self.cif_path), primitive=True)
                return self.structure
            except Exception:
                return None
        # Lazy load from stored structure (unified database) if available
        stored = self.generation_metadata.get("_stored_structure")
        if stored is not None:
            try:
                self.structure = stored.get_structure()
                return self.structure
            except Exception:
                return None
        return None

    def clear_structure(self) -> None:
        """Clear structure from memory (can be reloaded from CIF)."""
        self.structure = None


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
        database: Optional["StructureDatabase"] = None,
    ):
        """Initialize the chemistry explorer.

        Args:
            calculator: ASE calculator instance. If None, uses ORB calculator.
            random_seed: Optional random seed for reproducibility.
            output_dir: Base directory for storing results. If None, uses current directory.
            database: Optional unified StructureDatabase for cross-run structure sharing
                and global hull tracking. If provided, structures are automatically saved
                to both the per-run database and the unified database.
        """
        self._calculator = calculator  # Store provided calculator (may be None)
        self._calculator_initialized = calculator is not None
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        # Unified database for cross-run sharing
        self.database = database
        self._unified_run_id: Optional[str] = None

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

        # Remove duplicates but keep scaled versions that unlock different space groups.
        # Space groups have different minimum Wyckoff multiplicities (1, 2, 4, 8, etc.).
        # To access high-symmetry space groups, we need stoichiometries with counts
        # that are multiples of these multiplicities.
        #
        # For each unique composition ratio, keep versions at Z = 1, 2, 4, 8
        # (the key multiplicity tiers) if they fit within max_atoms.
        unique_by_reduced: Dict[str, List[Dict[str, int]]] = {}

        for stoich in stoichiometries:
            comp = Composition(stoich)
            reduced = comp.reduced_formula
            if reduced not in unique_by_reduced:
                unique_by_reduced[reduced] = []
            unique_by_reduced[reduced].append(stoich)

        # Key Z multipliers that unlock different space group tiers:
        # Z=1: SGs with min Wyckoff mult 1 (triclinic, most monoclinic)
        # Z=2: SGs with min mult 2 (many tetragonal, P4_2/mnm, etc.)
        # Z=4: SGs with min mult 4 (Fm-3m, etc.)
        # Z=8: SGs with min mult 8 (Fd-3m, etc.)
        key_multipliers = [1, 2, 4, 8]

        result = []
        for reduced, variants in unique_by_reduced.items():
            # Get the reduced composition as baseline
            comp = Composition(reduced)
            reduced_stoich = {
                str(el): int(comp.reduced_composition[el])
                for el in comp.reduced_composition.elements
            }
            base_total = sum(reduced_stoich.values())

            # Keep versions at key Z multipliers
            kept_totals = set()
            for z in key_multipliers:
                target_total = base_total * z
                if target_total > max_atoms:
                    break  # No point checking larger Z

                # Find a variant with this exact total
                for stoich in variants:
                    total = sum(stoich.values())
                    if total == target_total and total not in kept_totals:
                        result.append(stoich)
                        kept_totals.add(total)
                        break
                else:
                    # No exact match found, create the scaled version if it fits
                    if target_total <= max_atoms and target_total not in kept_totals:
                        scaled = {el: c * z for el, c in reduced_stoich.items()}
                        result.append(scaled)
                        kept_totals.add(target_total)

        return result

    def filter_stoichiometries_by_fraction(
        self,
        stoichiometries: List[Dict[str, int]],
        min_fraction: Optional[Dict[str, float]] = None,
        max_fraction: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, int]]:
        """Filter stoichiometries based on element fraction constraints.

        Args:
            stoichiometries: List of stoichiometry dictionaries.
            min_fraction: Minimum fraction constraints, e.g., {"Fe": 0.4} means
                Fe must be at least 40% of atoms.
            max_fraction: Maximum fraction constraints, e.g., {"Bi": 0.2} means
                Bi must be at most 20% of atoms.

        Returns:
            Filtered list of stoichiometries satisfying all constraints.

        Example:
            >>> stoichs = [{"Fe": 3, "Co": 1, "Bi": 1}, {"Fe": 1, "Co": 1, "Bi": 3}]
            >>> filtered = explorer.filter_stoichiometries_by_fraction(
            ...     stoichs, min_fraction={"Fe": 0.4}
            ... )
            >>> # Returns [{"Fe": 3, "Co": 1, "Bi": 1}] since Fe=3/5=60% >= 40%
        """
        if not min_fraction and not max_fraction:
            return stoichiometries

        min_fraction = min_fraction or {}
        max_fraction = max_fraction or {}

        filtered = []
        for stoich in stoichiometries:
            total_atoms = sum(stoich.values())
            if total_atoms == 0:
                continue

            passes = True

            # Check min_fraction constraints
            for element, min_frac in min_fraction.items():
                count = stoich.get(element, 0)
                actual_frac = count / total_atoms
                if actual_frac < min_frac:
                    passes = False
                    break

            # Check max_fraction constraints
            if passes:
                for element, max_frac in max_fraction.items():
                    count = stoich.get(element, 0)
                    actual_frac = count / total_atoms
                    if actual_frac > max_frac:
                        passes = False
                        break

            if passes:
                filtered.append(stoich)

        return filtered

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
                -- Dynamical stability (phonon) fields
                is_dynamically_stable INTEGER,
                num_imaginary_modes INTEGER,
                min_phonon_frequency REAL,
                max_phonon_frequency REAL,
                phonon_supercell TEXT,
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
        """Save a candidate to the per-run database and optionally to unified database."""
        # Convert metadata to JSON-serializable types
        metadata = self._to_json_serializable(candidate.generation_metadata)

        # Extract phonon stability info if available
        phonon_stability = candidate.generation_metadata.get("phonon_stability", {})
        is_dynamically_stable = phonon_stability.get("is_stable")
        num_imaginary_modes = phonon_stability.get("num_imaginary_modes")
        min_phonon_frequency = phonon_stability.get("min_frequency")
        max_phonon_frequency = phonon_stability.get("max_frequency")
        phonon_supercell = phonon_stability.get("supercell")

        cursor = conn.execute(
            """
            INSERT INTO candidates (
                formula, stoichiometry, energy_per_atom, total_energy,
                num_atoms, space_group_number, space_group_symbol,
                cif_filename, is_valid, error_message, generation_metadata,
                is_dynamically_stable, num_imaginary_modes, 
                min_phonon_frequency, max_phonon_frequency, phonon_supercell
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                (
                    1
                    if is_dynamically_stable
                    else (0 if is_dynamically_stable is False else None)
                ),
                num_imaginary_modes,
                min_phonon_frequency,
                max_phonon_frequency,
                json.dumps(phonon_supercell) if phonon_supercell else None,
            ),
        )
        conn.commit()

        # Also save to unified database if available
        if self.database is not None and candidate.is_valid:
            try:
                # Get CIF content
                cif_content = None
                structure = candidate.get_structure()
                if structure is not None:
                    try:
                        cif_content = structure.to(fmt="cif")
                    except Exception:
                        pass
                elif candidate.cif_path and candidate.cif_path.exists():
                    cif_content = candidate.cif_path.read_text()

                self.database.add_structure(
                    formula=candidate.formula,
                    stoichiometry=candidate.stoichiometry,
                    energy_per_atom=float(candidate.energy_per_atom),
                    total_energy=float(candidate.total_energy),
                    num_atoms=int(candidate.num_atoms),
                    space_group_number=int(candidate.space_group_number),
                    space_group_symbol=candidate.space_group_symbol,
                    cif_content=cif_content,
                    structure=structure,
                    is_valid=candidate.is_valid,
                    error_message=candidate.error_message,
                    generation_metadata=metadata,
                    run_id=self._unified_run_id,
                    is_dynamically_stable=is_dynamically_stable,
                    num_imaginary_modes=num_imaginary_modes,
                    min_phonon_frequency=min_phonon_frequency,
                    max_phonon_frequency=max_phonon_frequency,
                    phonon_supercell=phonon_supercell,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to save {candidate.formula} to unified database: {e}"
                )

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

    def _update_phonon_stability(
        self,
        conn: sqlite3.Connection,
        formula: str,
        is_stable: bool,
        num_imaginary_modes: int,
        min_frequency: Optional[float],
        max_frequency: Optional[float],
        supercell: Optional[Tuple[int, int, int]],
    ) -> None:
        """Update the phonon stability info for a candidate."""
        conn.execute(
            """
            UPDATE candidates
            SET is_dynamically_stable = ?,
                num_imaginary_modes = ?,
                min_phonon_frequency = ?,
                max_phonon_frequency = ?,
                phonon_supercell = ?
            WHERE formula = ?
            """,
            (
                1 if is_stable else 0,
                num_imaginary_modes,
                min_frequency,
                max_frequency,
                json.dumps(supercell) if supercell else None,
                formula,
            ),
        )
        conn.commit()

    def _calculate_phonon_stability(
        self,
        candidate: CandidateResult,
        supercell: Tuple[int, int, int] = (2, 2, 2),
        show_progress: bool = True,
    ) -> None:
        """Calculate phonon stability for a candidate and update its metadata.

        Args:
            candidate: CandidateResult to test.
            supercell: Supercell dimensions for phonon calculation.
            show_progress: Whether to show progress.
        """
        from .phonons import calculate_phonons

        structure = candidate.get_structure()
        if structure is None:
            return

        if show_progress:
            C = Colors
            print(f"    {C.DIM}Calculating phonons...{C.RESET}", end=" ", flush=True)

        try:
            phonon_result = calculate_phonons(
                structure=structure,
                calculator=self.calculator,
                supercell=supercell,
                generate_plot=False,
            )

            # Update candidate with stability info
            candidate.is_dynamically_stable = phonon_result.is_stable
            candidate.phonon_result = phonon_result
            candidate.generation_metadata["phonon_stability"] = {
                "is_stable": phonon_result.is_stable,
                "num_imaginary_modes": phonon_result.num_imaginary_modes,
                "min_frequency": phonon_result.min_frequency,
                "max_frequency": phonon_result.max_frequency,
                "min_imaginary_frequency": phonon_result.min_imaginary_frequency,
                "supercell": supercell,
            }

            if show_progress:
                C = Colors
                if phonon_result.is_stable:
                    print(f"{C.GREEN}✓ stable{C.RESET}")
                else:
                    print(
                        f"{C.RED}✗ unstable{C.RESET} "
                        f"({phonon_result.num_imaginary_modes} imag. modes)"
                    )

        except Exception as e:
            logger.warning(f"Phonon calculation failed for {candidate.formula}: {e}")
            candidate.generation_metadata["phonon_stability"] = {"error": str(e)}
            if show_progress:
                C = Colors
                print(f"{C.YELLOW}⚠ failed{C.RESET}")

    def _backfill_stability_for_hull_entries(
        self,
        candidates: List[CandidateResult],
        hull_entries: List[CandidateResult],
        conn: sqlite3.Connection,
        phonon_supercell: Tuple[int, int, int],
        e_above_hull_cutoff: float = 0.05,
        show_progress: bool = True,
    ) -> None:
        """Backfill phonon stability for hull and near-hull entries missing stability data.

        Args:
            candidates: All candidates from exploration.
            hull_entries: Candidates on the convex hull.
            conn: Database connection.
            phonon_supercell: Supercell dimensions for phonon calculation.
            e_above_hull_cutoff: Energy above hull cutoff for near-hull entries.
            show_progress: Whether to show progress.
        """
        # Identify candidates that need stability backfill
        # Include hull entries and near-hull entries
        candidates_to_backfill = []

        for candidate in candidates:
            if not candidate.is_valid:
                continue
            # Skip if already has stability data
            if candidate.is_dynamically_stable is not None:
                continue
            if "phonon_stability" in candidate.generation_metadata:
                if "error" not in candidate.generation_metadata["phonon_stability"]:
                    continue

            # Check if on hull or near hull
            e_above_hull = candidate.generation_metadata.get("e_above_hull")
            is_on_hull = candidate.generation_metadata.get("is_on_hull", False)

            if is_on_hull or (
                e_above_hull is not None and e_above_hull <= e_above_hull_cutoff
            ):
                candidates_to_backfill.append(candidate)

        if not candidates_to_backfill:
            return

        if show_progress:
            C = Colors
            print(
                f"\n{C.BOLD}Backfilling stability for {len(candidates_to_backfill)} "
                f"hull/near-hull entries...{C.RESET}"
            )

        for i, candidate in enumerate(candidates_to_backfill):
            if show_progress:
                C = Colors
                e_above = candidate.generation_metadata.get("e_above_hull", 0)
                print(
                    f"  {C.CYAN}[{i + 1}/{len(candidates_to_backfill)}]{C.RESET} "
                    f"{C.BOLD}{candidate.formula}{C.RESET} "
                    f"(E_hull={e_above:.3f} eV/atom)",
                    flush=True,
                )

            # Need to reload structure if cleared
            structure = candidate.get_structure()
            if structure is None:
                if show_progress:
                    C = Colors
                    print(f"    {C.YELLOW}⚠ Cannot load structure{C.RESET}")
                continue

            self._calculate_phonon_stability(
                candidate,
                supercell=phonon_supercell,
                show_progress=show_progress,
            )

            # Update database
            phonon_info = candidate.generation_metadata.get("phonon_stability", {})
            if "is_stable" in phonon_info:
                self._update_phonon_stability(
                    conn=conn,
                    formula=candidate.formula,
                    is_stable=phonon_info["is_stable"],
                    num_imaginary_modes=phonon_info.get("num_imaginary_modes", 0),
                    min_frequency=phonon_info.get("min_frequency"),
                    max_frequency=phonon_info.get("max_frequency"),
                    supercell=phonon_info.get("supercell"),
                )

    # -------------------- Structure Generation --------------------

    def _generate_structure_for_stoichiometry(
        self,
        stoichiometry: Dict[str, int],
        num_trials: int = 10,
        optimize: bool = True,
        symmetry_bias: float = 0.0,
        crystal_systems: Optional[List[str]] = None,
        space_group: Optional[int] = None,
        preserve_symmetry: bool = False,
        show_progress: bool = False,
        optimization_max_steps: int = 400,
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
            space_group: Optional specific space group number to target.
            preserve_symmetry: If True, use symmetry-constrained relaxation to preserve
                high symmetry during optimization.
            show_progress: If True, show tqdm progress bar during relaxation.

        Returns:
            CandidateResult with structure and energy information.
        """
        # Build formula string
        formula = "".join(
            f"{el}{count if count > 1 else ''}"
            for el, count in sorted(stoichiometry.items())
        )

        logger.debug(f"Generating structure for {formula}")

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
                space_group=space_group,
                num_trials=num_trials,
                optimize_geometry=optimize,
                multi_spacegroup=space_group
                is None,  # Disable multi if specific SG given
                top_k_spacegroups=5,
                symmetry_bias=symmetry_bias,
                crystal_systems=crystal_systems,
                refine_symmetry=not preserve_symmetry,  # Don't need refinement if preserving
                preserve_symmetry=preserve_symmetry,
                show_progress=show_progress,
                optimization_max_steps=optimization_max_steps,
            )

            structure = ggen.get_structure()
            if structure is None:
                raise ValueError("No structure generated")

            energy = result["best_crystal_energy"]
            num_atoms = len(structure)
            energy_per_atom = energy / num_atoms

            # Save additional relaxed trials (polymorphs) to unified database
            # These are valuable structures that were fully relaxed but weren't the best
            additional_trials = result.get("all_relaxed_trials", [])
            if additional_trials and self.database is not None:
                for trial in additional_trials:
                    try:
                        trial_struct = trial["structure"]
                        trial_cif = trial_struct.to(fmt="cif")
                        self.database.add_structure(
                            formula=formula,
                            stoichiometry=stoichiometry,
                            energy_per_atom=float(trial["energy_per_atom"]),
                            total_energy=float(trial["energy"]),
                            num_atoms=int(trial["num_atoms"]),
                            space_group_number=int(trial["space_group_number"]),
                            space_group_symbol=trial["space_group_symbol"],
                            cif_content=trial_cif,
                            structure=None,  # Don't store structure object to save memory
                            is_valid=True,
                            generation_metadata={
                                "is_additional_trial": True,
                                "original_space_group": trial.get(
                                    "original_space_group"
                                ),
                                "optimization_steps": trial.get(
                                    "optimization_steps", 0
                                ),
                            },
                            run_id=self._unified_run_id,
                        )
                        # Clear structure from trial dict to free memory
                        trial["structure"] = None
                    except Exception as e:
                        logger.debug(
                            f"Failed to save additional trial for {formula}: {e}"
                        )

                if additional_trials:
                    logger.debug(
                        f"Saved {len(additional_trials)} additional polymorphs for {formula}"
                    )

            # Track count before clearing
            num_trials_saved = len(additional_trials)

            # Clear all_relaxed_trials from result to free memory
            # (we've already saved them to the database and set structures to None)
            result["all_relaxed_trials"] = None
            del additional_trials

            # Clean up the GGen instance to free memory
            ggen.cleanup()

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
                    "num_additional_trials_saved": num_trials_saved,
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

        # Get the structure (may load from CIF if cleared from memory)
        structure = candidate.get_structure()
        if structure is None:
            raise ValueError(f"No structure available for {candidate.formula}")

        # Get primitive cell to avoid supercell issues on reload
        # This ensures consistent atom counts between save and load
        try:
            structure = structure.get_primitive_structure()
        except Exception:
            # Fall back to original structure if primitive conversion fails
            pass

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
        show_progress: bool = False,
        optimization_max_steps: int = 400,
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
            show_progress: If True, show tqdm progress bar during relaxation.

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
                        show_progress=show_progress,
                        optimization_max_steps=optimization_max_steps,
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
        # A candidate is valid for phase diagram if it has valid energy data
        # (structure may be cleared from memory but data is preserved)
        valid_candidates = [
            c
            for c in candidates
            if c.is_valid and (c.structure is not None or c.cif_path is not None)
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
            if terminal.is_valid and (
                terminal.structure is not None or terminal.cif_path is not None
            ):
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

    # -------------------- Parallel Generation --------------------

    def _generate_parallel(
        self,
        stoichs_to_generate: List[Tuple[Dict[str, int], str]],
        previous_structures: Dict[str, CandidateResult],
        candidates: List[CandidateResult],
        num_successful: int,
        num_failed: int,
        structures_dir: Path,
        conn: sqlite3.Connection,
        num_trials: int,
        optimize: bool,
        symmetry_bias: float,
        crystal_systems: Optional[List[str]],
        space_group: Optional[int],
        preserve_symmetry: bool,
        num_workers: int,
        show_progress: bool,
        keep_structures_in_memory: bool,
        interrupted_flag,
        compute_phonons: bool = False,
        phonon_supercell: Tuple[int, int, int] = (2, 2, 2),
        optimization_max_steps: int = 400,
    ) -> Tuple[List[CandidateResult], int, int]:
        """Generate structures in parallel using ProcessPoolExecutor.

        Returns updated (candidates, num_successful, num_failed).
        """
        # Prepare worker arguments
        worker_args = []
        for stoich, formula in stoichs_to_generate:
            worker_args.append(
                {
                    "stoichiometry": stoich,
                    "num_trials": num_trials,
                    "optimize": optimize,
                    "symmetry_bias": symmetry_bias,
                    "crystal_systems": crystal_systems,
                    "space_group": space_group,
                    "preserve_symmetry": preserve_symmetry,
                    "random_seed": self.random_seed,
                    "optimization_max_steps": optimization_max_steps,
                }
            )

        # Use ProcessPoolExecutor for parallel generation
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(_generate_structure_worker, args): (
                    args,
                    stoichs_to_generate[i][1],
                )
                for i, args in enumerate(worker_args)
            }

            # Process results as they complete with progress bar
            pbar = tqdm(
                as_completed(future_to_args),
                total=len(future_to_args),
                desc="Generating structures",
                disable=not show_progress,
                unit="struct",
            )

            for future in pbar:
                if interrupted_flag():
                    logger.info("Interrupt detected - cancelling remaining tasks")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                args, formula = future_to_args[future]
                pbar.set_postfix_str(formula)

                try:
                    result_dict = future.result()

                    # Convert dict to CandidateResult
                    candidate = CandidateResult(
                        formula=result_dict["formula"],
                        stoichiometry=result_dict["stoichiometry"],
                        energy_per_atom=result_dict["energy_per_atom"],
                        total_energy=result_dict["total_energy"],
                        num_atoms=result_dict["num_atoms"],
                        space_group_number=result_dict["space_group_number"],
                        space_group_symbol=result_dict["space_group_symbol"],
                        structure=result_dict["structure"],
                        generation_metadata=result_dict["generation_metadata"],
                        is_valid=result_dict["is_valid"],
                        error_message=result_dict["error_message"],
                    )

                    # Save additional relaxed trials (polymorphs) to unified database
                    additional_trials = result_dict.get("all_relaxed_trials", [])
                    if additional_trials and self.database is not None:
                        stoich = result_dict["stoichiometry"]
                        for trial in additional_trials:
                            try:
                                trial_struct = trial["structure"]
                                trial_cif = trial_struct.to(fmt="cif")
                                self.database.add_structure(
                                    formula=result_dict["formula"],
                                    stoichiometry=stoich,
                                    energy_per_atom=float(trial["energy_per_atom"]),
                                    total_energy=float(trial["energy"]),
                                    num_atoms=int(trial["num_atoms"]),
                                    space_group_number=int(trial["space_group_number"]),
                                    space_group_symbol=trial["space_group_symbol"],
                                    cif_content=trial_cif,
                                    structure=None,  # Don't store structure object to save memory
                                    is_valid=True,
                                    generation_metadata={
                                        "is_additional_trial": True,
                                        "original_space_group": trial.get(
                                            "original_space_group"
                                        ),
                                        "optimization_steps": trial.get(
                                            "optimization_steps", 0
                                        ),
                                    },
                                    run_id=self._unified_run_id,
                                )
                                # Clear structure from trial dict to free memory
                                trial["structure"] = None
                            except Exception as e:
                                logger.debug(
                                    f"Failed to save additional trial for {formula}: {e}"
                                )

                        if additional_trials:
                            logger.debug(
                                f"Saved {len(additional_trials)} additional polymorphs for {formula}"
                            )

                    # Clear the trials list from result dict to free memory
                    result_dict["all_relaxed_trials"] = None

                    # If we have a previous structure and this one failed or is worse, use the previous
                    if formula in previous_structures:
                        prev_candidate = previous_structures[formula]
                        if not candidate.is_valid or (
                            prev_candidate.is_valid
                            and prev_candidate.energy_per_atom
                            < candidate.energy_per_atom
                        ):
                            logger.debug(
                                f"Using better structure from previous run for {formula}"
                            )
                            candidate = prev_candidate
                            candidate.generation_metadata["reused_from_previous"] = True

                    if candidate.is_valid and candidate.structure is not None:
                        # Save CIF
                        cif_path = self._save_structure_cif(candidate, structures_dir)
                        candidate.cif_path = cif_path
                        num_successful += 1

                        # Calculate phonon stability if enabled
                        if compute_phonons:
                            self._calculate_phonon_stability(
                                candidate,
                                supercell=phonon_supercell,
                                show_progress=False,  # Don't show individual progress in parallel mode
                            )

                        # Clear structure from memory if not needed
                        if not keep_structures_in_memory:
                            candidate.clear_structure()
                    else:
                        num_failed += 1

                    # Save to database
                    self._save_candidate(conn, candidate)
                    candidates.append(candidate)

                    # Aggressive memory cleanup after each stoichiometry
                    # to prevent memory accumulation during long runs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.warning(f"Worker failed for {formula}: {e}")
                    # Create a failed candidate
                    candidate = CandidateResult(
                        formula=formula,
                        stoichiometry=args["stoichiometry"],
                        energy_per_atom=float("nan"),
                        total_energy=float("nan"),
                        num_atoms=0,
                        space_group_number=0,
                        space_group_symbol="",
                        structure=None,
                        is_valid=False,
                        error_message=str(e),
                    )
                    self._save_candidate(conn, candidate)
                    candidates.append(candidate)
                    num_failed += 1

            pbar.close()

        return candidates, num_successful, num_failed

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
        space_group: Optional[int] = None,
        load_previous_runs: Union[bool, int] = False,
        skip_existing_formulas: bool = True,
        preserve_symmetry: bool = False,
        num_workers: int = 1,
        show_progress: bool = True,
        keep_structures_in_memory: bool = False,
        use_unified_database: bool = True,
        compute_phonons: bool = False,
        phonon_supercell: Tuple[int, int, int] = (2, 2, 2),
        min_fraction: Optional[Dict[str, float]] = None,
        max_fraction: Optional[Dict[str, float]] = None,
        optimization_max_steps: int = 400,
    ) -> ExplorationResult:
        """Explore a chemical system by generating candidate structures.

        This is the main entry point for chemistry exploration. It:
        1. Parses the chemical system
        2. Optionally loads structures from previous runs
        3. Enumerates candidate stoichiometries
        4. Generates and optimizes structures for each (skipping already-explored ones)
        5. Stores all data in SQLite and CIF files
        6. Builds a phase diagram
        7. Optionally tests dynamical stability via phonon calculations
        8. Returns stable candidates

        Supports graceful interruption with Ctrl+C - partial results will be saved.

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
            space_group: Optional specific space group number (1-230) to target.
                If provided, only this space group will be used for generation.
                Example: 136 for P4_2/mnm (tetragonal sigma phase).
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
            num_workers: Number of parallel workers for structure generation. Default: 1
                (sequential). Set > 1 for parallel generation. Note: each worker loads
                its own ML model, so memory usage scales with num_workers.
            show_progress: Whether to show tqdm progress bar. Default: True.
            keep_structures_in_memory: If False (default), structures are saved to CIF
                and cleared from memory to reduce memory usage. Set True to keep all
                structures in memory (useful for small explorations or post-processing).
            use_unified_database: If True (default) and a unified StructureDatabase is
                configured, load existing structures from all relevant subsystems. For
                example, when exploring Fe-Mn-Co, this will load existing Fe, Mn, Co,
                Fe-Mn, Fe-Co, Mn-Co structures. This is more powerful than load_previous_runs
                as it shares structures across different chemical systems.
            compute_phonons: If True, compute phonon stability for each generated structure.
                Default False - phonon calculations are expensive and can be run later
                using the phonons.py script for structures of interest.
            phonon_supercell: Supercell dimensions for phonon calculations (if enabled).
                Default (2,2,2).
            min_fraction: Minimum element fraction constraints. Dict mapping element symbols
                to minimum fractions (0.0-1.0). E.g., {"Fe": 0.4} means Fe must be at least
                40% of atoms in each stoichiometry. Useful for biasing exploration toward
                compositions rich in specific elements.
            max_fraction: Maximum element fraction constraints. Dict mapping element symbols
                to maximum fractions (0.0-1.0). E.g., {"Bi": 0.2} means Bi can be at most
                20% of atoms. Can be combined with min_fraction.

        Returns:
            ExplorationResult with all candidates, phase diagram, and stable phases.
        """
        import time

        # Suppress pymatgen CIF parsing warnings
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="pymatgen.core.structure"
        )
        # Suppress pymatgen CIF stoichiometry warnings
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="pymatgen.io.cif"
        )
        # Suppress orb_models torch dtype warnings
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="orb_models.utils"
        )

        start_time = time.time()

        # Setup interrupt handling for graceful shutdown
        interrupted = False
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def handle_interrupt(signum, frame):
            nonlocal interrupted
            if interrupted:
                # Second Ctrl+C - force exit
                logger.warning("Force interrupt - exiting immediately")
                raise KeyboardInterrupt
            interrupted = True
            logger.warning(
                "\nInterrupt received - finishing current structure and saving results..."
            )

        signal.signal(signal.SIGINT, handle_interrupt)

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

        # Create run in unified database if available
        if self.database is not None:
            self._unified_run_id = self.database.create_run(
                chemical_system=chemsys,
                parameters={
                    "max_atoms": max_atoms,
                    "min_atoms": min_atoms,
                    "num_trials": num_trials,
                    "optimize": optimize,
                    "include_binaries": include_binaries,
                    "include_ternaries": include_ternaries,
                    "require_all_elements": require_all_elements,
                    "symmetry_bias": symmetry_bias,
                    "crystal_systems": crystal_systems,
                    "preserve_symmetry": preserve_symmetry,
                    "num_workers": num_workers,
                },
            )
            logger.info(f"Created unified database run: {self._unified_run_id[:8]}...")

        # Load structures from unified database if available
        previous_structures: Dict[str, CandidateResult] = {}
        if use_unified_database and self.database is not None:
            unified_structures = self._load_from_unified_database(
                chemsys, keep_structures_in_memory=keep_structures_in_memory
            )
            if unified_structures:
                previous_structures.update(unified_structures)
                logger.info(
                    f"Loaded {len(unified_structures)} structures from unified database"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
                    (
                        "loaded_from_unified_db",
                        json.dumps(list(unified_structures.keys())),
                    ),
                )
                conn.commit()

        # Load structures from previous runs if requested (legacy mode)
        if load_previous_runs:
            max_runs = None if load_previous_runs is True else int(load_previous_runs)
            run_structures = self.load_structures_from_previous_runs(
                chemical_system=chemsys,
                max_runs=max_runs,
            )
            # Merge, keeping lower energy structures
            for formula, candidate in run_structures.items():
                if formula not in previous_structures:
                    previous_structures[formula] = candidate
                elif (
                    candidate.energy_per_atom
                    < previous_structures[formula].energy_per_atom
                ):
                    previous_structures[formula] = candidate
            conn.execute(
                "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
                ("loaded_from_previous", json.dumps(list(run_structures.keys()))),
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

        # Filter stoichiometries by element fraction constraints if specified
        if min_fraction or max_fraction:
            original_count = len(stoichiometries)
            stoichiometries = self.filter_stoichiometries_by_fraction(
                stoichiometries,
                min_fraction=min_fraction,
                max_fraction=max_fraction,
            )
            filtered_count = original_count - len(stoichiometries)
            if filtered_count > 0:
                # Build constraint description for logging
                constraints = []
                if min_fraction:
                    for el, frac in min_fraction.items():
                        constraints.append(f"{el} ≥ {frac*100:.1f}%")
                if max_fraction:
                    for el, frac in max_fraction.items():
                        constraints.append(f"{el} ≤ {frac*100:.1f}%")
                logger.info(
                    f"Filtered {filtered_count} stoichiometries not matching "
                    f"composition constraints ({', '.join(constraints)}). "
                    f"{len(stoichiometries)} remaining."
                )

        # Filter stoichiometries by space group compatibility if specified.
        # enumerate_stoichiometries now returns versions at Z=1,2,4,8 for each
        # composition ratio, so we just need to check exact counts.
        if space_group is not None:
            from pyxtal.symmetry import Group

            group = Group(space_group, dim=3)
            original_count = len(stoichiometries)

            # Cache: map from tuple(counts) -> bool for already-checked counts
            compatibility_cache: Dict[Tuple[int, ...], bool] = {}

            compatible_stoichs = []
            for stoich in stoichiometries:
                counts = tuple(stoich[el] for el in sorted(stoich.keys()))

                # Check cache first
                if counts in compatibility_cache:
                    if compatibility_cache[counts]:
                        compatible_stoichs.append(stoich)
                    continue

                is_compatible, _ = group.check_compatible(list(counts))
                compatibility_cache[counts] = is_compatible
                if is_compatible:
                    compatible_stoichs.append(stoich)

            stoichiometries = compatible_stoichs
            filtered_count = original_count - len(stoichiometries)
            if filtered_count > 0:
                logger.info(
                    f"Filtered {filtered_count} stoichiometries incompatible with "
                    f"space group {space_group} ({len(stoichiometries)} remaining)"
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

        # Separate stoichiometries into reused vs need-to-generate
        stoichs_to_generate = []
        for stoich in stoichiometries:
            formula = "".join(
                f"{el}{count if count > 1 else ''}"
                for el, count in sorted(stoich.items())
            )

            # Check if we already have this formula from previous runs
            if formula in previous_structures and skip_existing_formulas:
                prev_candidate = previous_structures[formula]
                logger.debug(
                    f"Reusing {formula} from previous run "
                    f"(E={prev_candidate.energy_per_atom:.4f} eV/atom)"
                )

                # Copy the structure to the new run's structures directory
                structure = prev_candidate.get_structure()
                if structure is not None:
                    cif_path = self._save_structure_cif(prev_candidate, structures_dir)
                    prev_candidate.cif_path = cif_path
                    if not keep_structures_in_memory:
                        prev_candidate.clear_structure()

                # Mark as reused in metadata
                prev_candidate.generation_metadata["reused_from_previous"] = True

                # Save to database
                self._save_candidate(conn, prev_candidate)
                candidates.append(prev_candidate)
                num_successful += 1
                num_reused += 1
            else:
                stoichs_to_generate.append((stoich, formula))

        if num_reused > 0:
            logger.info(f"Reused {num_reused} structures from previous runs")

        # Generate new structures (sequential or parallel)
        if stoichs_to_generate and not interrupted:
            if num_workers > 1:
                # Parallel generation
                logger.info(
                    f"Generating {len(stoichs_to_generate)} structures "
                    f"with {num_workers} workers"
                )
                candidates, num_successful, num_failed = self._generate_parallel(
                    stoichs_to_generate=stoichs_to_generate,
                    previous_structures=previous_structures,
                    candidates=candidates,
                    num_successful=num_successful,
                    num_failed=num_failed,
                    structures_dir=structures_dir,
                    conn=conn,
                    num_trials=num_trials,
                    optimize=optimize,
                    symmetry_bias=symmetry_bias,
                    crystal_systems=crystal_systems,
                    space_group=space_group,
                    preserve_symmetry=preserve_symmetry,
                    num_workers=num_workers,
                    show_progress=show_progress,
                    keep_structures_in_memory=keep_structures_in_memory,
                    interrupted_flag=lambda: interrupted,
                    compute_phonons=compute_phonons,
                    phonon_supercell=phonon_supercell,
                    optimization_max_steps=optimization_max_steps,
                )
            else:
                # Sequential generation - show relaxation progress for each structure
                for i, (stoich, formula) in enumerate(stoichs_to_generate):
                    if interrupted:
                        logger.info("Stopping generation due to interrupt")
                        break

                    if show_progress:
                        C = Colors
                        print(
                            f"{C.CYAN}[{i + 1}/{len(stoichs_to_generate)}]{C.RESET} "
                            f"{C.BOLD}{formula}{C.RESET}",
                            flush=True,
                        )

                    candidate = self._generate_structure_for_stoichiometry(
                        stoichiometry=stoich,
                        num_trials=num_trials,
                        optimize=optimize,
                        symmetry_bias=symmetry_bias,
                        crystal_systems=crystal_systems,
                        space_group=space_group,
                        preserve_symmetry=preserve_symmetry,
                        show_progress=show_progress,
                        optimization_max_steps=optimization_max_steps,
                    )

                    # If we have a previous structure and this one failed or is worse, use the previous
                    if formula in previous_structures:
                        prev_candidate = previous_structures[formula]
                        if not candidate.is_valid or (
                            prev_candidate.is_valid
                            and prev_candidate.energy_per_atom
                            < candidate.energy_per_atom
                        ):
                            logger.debug(
                                f"Using better structure from previous run for {formula} "
                                f"(prev={prev_candidate.energy_per_atom:.4f} vs "
                                f"new={candidate.energy_per_atom:.4f} eV/atom)"
                            )
                            candidate = prev_candidate
                            candidate.generation_metadata["reused_from_previous"] = True

                    if candidate.is_valid:
                        # Load structure if needed (lazy loading from database)
                        structure = candidate.get_structure()
                        if structure is not None:
                            # Save CIF
                            cif_path = self._save_structure_cif(candidate, structures_dir)
                            candidate.cif_path = cif_path
                            num_successful += 1

                            # Calculate phonon stability if enabled
                            if compute_phonons:
                                self._calculate_phonon_stability(
                                    candidate,
                                    supercell=phonon_supercell,
                                    show_progress=show_progress,
                                )

                            # Clear structure from memory if not needed
                            if not keep_structures_in_memory:
                                candidate.clear_structure()
                            # Remove stored structure reference to free memory
                            candidate.generation_metadata.pop("_stored_structure", None)
                        else:
                            num_failed += 1
                    else:
                        num_failed += 1

                    # Save to database
                    self._save_candidate(conn, candidate)
                    candidates.append(candidate)

                    # Aggressive memory cleanup after each stoichiometry
                    # to prevent memory accumulation during long runs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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
                    structure = prev_candidate.get_structure()
                    if structure is not None:
                        cif_path = self._save_structure_cif(
                            prev_candidate, structures_dir
                        )
                        prev_candidate.cif_path = cif_path
                        if not keep_structures_in_memory:
                            prev_candidate.clear_structure()
                        # Remove stored structure reference to free memory
                        prev_candidate.generation_metadata.pop("_stored_structure", None)

                    prev_candidate.generation_metadata["reused_from_previous"] = True
                    self._save_candidate(conn, prev_candidate)
                    candidates.append(prev_candidate)
                    num_successful += 1
                    num_reused += 1

            logger.info(
                f"Included {len(candidates)} total candidates "
                f"({num_reused} from previous runs)"
            )

        # Clear previous_structures to free memory now that all processing is complete
        previous_structures.clear()
        gc.collect()

        # Generate terminal element structures for phase diagram (unless interrupted)
        # Always generate fresh terminals and compare with previous runs to get the best
        terminal_candidates = []

        # Sanity bounds for terminal element energies (eV/atom)
        # Most metals are -4 to -10, some heavier metals up to -12
        # Anything outside this range is likely a calculation error
        MIN_REASONABLE_ENERGY = -12.0  # More negative than this is suspicious
        MAX_REASONABLE_ENERGY = -1.0  # Less negative than this is suspicious

        def _is_reasonable_terminal_energy(energy: float) -> bool:
            return MIN_REASONABLE_ENERGY <= energy <= MAX_REASONABLE_ENERGY

        if not interrupted:
            logger.info(f"Generating terminal element structures for {elements}")
            new_terminals = self._generate_terminal_elements(
                elements=elements,
                num_trials=num_trials,
                optimize=optimize,
                symmetry_bias=symmetry_bias,
                preserve_symmetry=preserve_symmetry,
                show_progress=show_progress,
                optimization_max_steps=optimization_max_steps,
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
                structure = terminal.get_structure()
                if terminal.is_valid and structure is not None:
                    cif_path = self._save_structure_cif(terminal, structures_dir)
                    terminal.cif_path = cif_path
                    self._save_candidate(conn, terminal)
                    if not keep_structures_in_memory:
                        terminal.clear_structure()

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

                # Update hull in unified database
                if self.database is not None:
                    try:
                        self.database.compute_hull(chemsys, update_database=True)
                        logger.info(
                            f"Updated global hull for {chemsys} in unified database"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update unified database hull: {e}")

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

        # Backfill: Check hull/near-hull entries for missing stability data
        if compute_phonons and not interrupted and hull_entries:
            self._backfill_stability_for_hull_entries(
                candidates=candidates,
                hull_entries=hull_entries,
                conn=conn,
                phonon_supercell=phonon_supercell,
                e_above_hull_cutoff=0.05,  # Also check near-hull entries
                show_progress=show_progress,
            )

        # Finalize
        total_time = time.time() - start_time

        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)

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
        conn.execute(
            "INSERT OR REPLACE INTO run_metadata VALUES (?, ?)",
            ("interrupted", "1" if interrupted else "0"),
        )
        conn.commit()
        conn.close()

        # Complete run in unified database
        if self.database is not None and self._unified_run_id is not None:
            self.database.complete_run(
                run_id=self._unified_run_id,
                num_candidates=len(candidates),
                num_successful=num_successful,
                num_failed=num_failed,
            )

        reused_msg = f" ({num_reused} from previous runs)" if num_reused > 0 else ""
        interrupted_msg = " (INTERRUPTED)" if interrupted else ""
        logger.info(
            f"Exploration{interrupted_msg}: {num_successful} successful{reused_msg}, "
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
                    # Check if candidate has a loadable structure (in memory or on disk)
                    if not candidate.is_valid:
                        continue
                    if candidate.structure is None and candidate.cif_path is None:
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

    def _load_from_unified_database(
        self, chemical_system: str, keep_structures_in_memory: bool = False
    ) -> Dict[str, CandidateResult]:
        """Load structures from unified database for a chemical system and all subsystems.

        Args:
            chemical_system: Chemical system to load (e.g., "Fe-Mn-Co")
            keep_structures_in_memory: If False, structures are not loaded into memory
                to save memory. They will be loaded lazily when needed.

        Returns:
            Dictionary mapping formula -> CandidateResult with best structures
            from all relevant subsystems.
        """
        if self.database is None:
            return {}

        best_structures = self.database.get_best_structures_for_subsystem(
            chemical_system
        )

        # Convert StoredStructure to CandidateResult
        results: Dict[str, CandidateResult] = {}
        for formula, stored in best_structures.items():
            # Only load structure into memory if explicitly requested
            # Otherwise, store CIF content reference for lazy loading
            structure = None
            if keep_structures_in_memory:
                structure = stored.get_structure()
            
            # Create a wrapper that can lazily load the structure when needed
            # Store the StoredStructure reference for lazy loading
            candidate = CandidateResult(
                formula=stored.formula,
                stoichiometry=stored.stoichiometry,
                structure=structure,
                energy_per_atom=stored.energy_per_atom,
                total_energy=stored.total_energy,
                num_atoms=stored.num_atoms,
                space_group_number=stored.space_group_number or 1,
                space_group_symbol=stored.space_group_symbol or "P1",
                cif_path=None,  # Will be set when saved to new run
                is_valid=stored.is_valid,
                error_message=stored.error_message,
                generation_metadata={
                    **stored.generation_metadata,
                    "loaded_from_unified_db": True,
                    "unified_db_id": stored.id,
                    "e_above_hull": stored.e_above_hull,
                    "is_on_hull": stored.is_on_hull,
                    "_stored_structure": stored,  # Store reference for lazy loading
                },
            )
            results[formula] = candidate

        return results

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

    # -------------------- Dynamical Stability Testing --------------------

    def test_dynamical_stability(
        self,
        result: ExplorationResult,
        candidates_to_test: Optional[List[CandidateResult]] = None,
        supercell: Tuple[int, int, int] = (2, 2, 2),
        test_hull_only: bool = True,
        max_candidates: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Test candidates for dynamical stability using phonon calculations.

        This method computes phonon dispersion for selected candidates and
        checks for imaginary modes that indicate dynamical instability.

        Args:
            result: ExplorationResult from explore().
            candidates_to_test: Specific candidates to test. If None, uses
                hull entries (if test_hull_only=True) or all valid candidates.
            supercell: Supercell dimensions for phonon calculation.
                Larger = more accurate but slower. (2,2,2) is usually sufficient.
            test_hull_only: If True and candidates_to_test is None, only test
                candidates on the convex hull.
            max_candidates: Maximum number of candidates to test.
            show_progress: Whether to show progress information.

        Returns:
            Dictionary with:
            - 'tested': List of tested candidates with stability info
            - 'stable': List of dynamically stable candidates
            - 'unstable': List of dynamically unstable candidates
            - 'failed': List of candidates where phonon calc failed
            - 'summary': Summary statistics

        Raises:
            ImportError: If phonopy is not installed.
        """
        from .phonons import calculate_phonons, PhononResult

        # Select candidates to test
        if candidates_to_test is not None:
            to_test = candidates_to_test
        elif test_hull_only and result.hull_entries:
            to_test = result.hull_entries
        else:
            to_test = [c for c in result.candidates if c.is_valid]

        # Apply max limit
        if max_candidates is not None:
            to_test = to_test[:max_candidates]

        if not to_test:
            logger.warning("No candidates to test for dynamical stability")
            return {
                "tested": [],
                "stable": [],
                "unstable": [],
                "failed": [],
                "summary": {"num_tested": 0},
            }

        stable = []
        unstable = []
        failed = []

        if show_progress:
            C = Colors
            print(
                f"\n{C.BOLD}Testing {len(to_test)} candidates for dynamical stability...{C.RESET}"
            )

        for i, candidate in enumerate(to_test):
            structure = candidate.get_structure()
            if structure is None:
                logger.warning(f"Cannot load structure for {candidate.formula}")
                failed.append(candidate)
                continue

            if show_progress:
                C = Colors
                print(
                    f"  {C.CYAN}[{i + 1}/{len(to_test)}]{C.RESET} "
                    f"{C.BOLD}{candidate.formula}{C.RESET} "
                    f"(E={candidate.energy_per_atom:.4f} eV/atom, SG={candidate.space_group_symbol})",
                    flush=True,
                )

            try:
                phonon_result = calculate_phonons(
                    structure=structure,
                    calculator=self.calculator,
                    supercell=supercell,
                    generate_plot=False,
                )

                # Update candidate with stability info
                candidate.is_dynamically_stable = phonon_result.is_stable
                candidate.phonon_result = phonon_result
                candidate.generation_metadata["phonon_stability"] = {
                    "is_stable": phonon_result.is_stable,
                    "num_imaginary_modes": phonon_result.num_imaginary_modes,
                    "min_frequency": phonon_result.min_frequency,
                    "max_frequency": phonon_result.max_frequency,
                    "min_imaginary_frequency": phonon_result.min_imaginary_frequency,
                    "supercell": supercell,
                }

                if phonon_result.is_stable:
                    stable.append(candidate)
                    if show_progress:
                        print(f"    {C.GREEN}✓ Dynamically stable{C.RESET}")
                else:
                    unstable.append(candidate)
                    if show_progress:
                        print(
                            f"    {C.RED}✗ Unstable{C.RESET} "
                            f"({phonon_result.num_imaginary_modes} imaginary modes, "
                            f"min freq: {phonon_result.min_imaginary_frequency:.3f} THz)"
                        )

            except Exception as e:
                logger.warning(
                    f"Phonon calculation failed for {candidate.formula}: {e}"
                )
                candidate.is_dynamically_stable = None
                candidate.generation_metadata["phonon_stability"] = {"error": str(e)}
                failed.append(candidate)
                if show_progress:
                    print(f"    {C.YELLOW}⚠ Calculation failed: {e}{C.RESET}")

        summary = {
            "num_tested": len(to_test),
            "num_stable": len(stable),
            "num_unstable": len(unstable),
            "num_failed": len(failed),
            "supercell": supercell,
        }

        if show_progress:
            C = Colors
            print(f"\n{C.BOLD}Stability Summary:{C.RESET}")
            print(f"  Stable:   {C.GREEN}{len(stable)}{C.RESET}")
            print(f"  Unstable: {C.RED}{len(unstable)}{C.RESET}")
            if failed:
                print(f"  Failed:   {C.YELLOW}{len(failed)}{C.RESET}")

        return {
            "tested": to_test,
            "stable": stable,
            "unstable": unstable,
            "failed": failed,
            "summary": summary,
        }

    def find_stable_hull_candidates(
        self,
        result: ExplorationResult,
        supercell: Tuple[int, int, int] = (2, 2, 2),
        max_alternatives: int = 3,
        show_progress: bool = True,
    ) -> Dict[str, CandidateResult]:
        """Find dynamically stable candidates for each composition on the hull.

        For each composition on the convex hull, tests the best candidate for
        stability. If unstable, tests alternatives (next-best by energy) until
        a stable candidate is found or max_alternatives is reached.

        Args:
            result: ExplorationResult from explore().
            supercell: Supercell dimensions for phonon calculation.
            max_alternatives: Maximum alternative candidates to test per composition
                if the best is unstable.
            show_progress: Whether to show progress information.

        Returns:
            Dictionary mapping formula -> best stable CandidateResult for that formula.
            Only includes formulas where a stable candidate was found.
        """
        from .phonons import calculate_phonons

        if not result.hull_entries:
            logger.warning("No hull entries to test")
            return {}

        # Group all candidates by formula
        candidates_by_formula: Dict[str, List[CandidateResult]] = {}
        for candidate in result.candidates:
            if candidate.is_valid and candidate.get_structure() is not None:
                formula = candidate.formula
                if formula not in candidates_by_formula:
                    candidates_by_formula[formula] = []
                candidates_by_formula[formula].append(candidate)

        # Sort each formula's candidates by energy
        for formula in candidates_by_formula:
            candidates_by_formula[formula].sort(key=lambda c: c.energy_per_atom)

        # Get unique formulas on the hull
        hull_formulas = set(c.formula for c in result.hull_entries)

        stable_candidates: Dict[str, CandidateResult] = {}

        if show_progress:
            C = Colors
            print(
                f"\n{C.BOLD}Finding stable candidates for {len(hull_formulas)} hull compositions...{C.RESET}\n"
            )

        for formula in hull_formulas:
            candidates = candidates_by_formula.get(formula, [])
            if not candidates:
                continue

            num_to_test = min(len(candidates), max_alternatives + 1)

            if show_progress:
                C = Colors
                print(f"{C.BOLD}{formula}{C.RESET} ({num_to_test} candidates to test)")

            for i, candidate in enumerate(candidates[:num_to_test]):
                structure = candidate.get_structure()
                if structure is None:
                    continue

                if show_progress:
                    rank_label = "best" if i == 0 else f"#{i + 1}"
                    print(
                        f"  Testing {rank_label}: E={candidate.energy_per_atom:.4f} eV/atom, "
                        f"SG={candidate.space_group_symbol}",
                        end=" ",
                        flush=True,
                    )

                try:
                    phonon_result = calculate_phonons(
                        structure=structure,
                        calculator=self.calculator,
                        supercell=supercell,
                        generate_plot=False,
                    )

                    candidate.is_dynamically_stable = phonon_result.is_stable
                    candidate.phonon_result = phonon_result
                    candidate.generation_metadata["phonon_stability"] = {
                        "is_stable": phonon_result.is_stable,
                        "num_imaginary_modes": phonon_result.num_imaginary_modes,
                        "min_frequency": phonon_result.min_frequency,
                        "min_imaginary_frequency": phonon_result.min_imaginary_frequency,
                    }

                    if phonon_result.is_stable:
                        stable_candidates[formula] = candidate
                        if show_progress:
                            C = Colors
                            print(f"{C.GREEN}✓ STABLE{C.RESET}")
                        break  # Found stable candidate for this formula
                    else:
                        if show_progress:
                            C = Colors
                            print(
                                f"{C.RED}✗ unstable{C.RESET} "
                                f"({phonon_result.num_imaginary_modes} imag. modes)"
                            )

                except Exception as e:
                    if show_progress:
                        C = Colors
                        print(f"{C.YELLOW}⚠ failed{C.RESET}")
                    logger.warning(f"Phonon calculation failed: {e}")

            if formula not in stable_candidates and show_progress:
                C = Colors
                print(f"  {C.YELLOW}No stable candidate found for {formula}{C.RESET}")

            if show_progress:
                print()  # Blank line between formulas

        if show_progress:
            C = Colors
            print(
                f"{C.BOLD}Found {len(stable_candidates)}/{len(hull_formulas)} stable hull compositions{C.RESET}"
            )

        return stable_candidates
