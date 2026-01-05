from __future__ import annotations

import base64
import io
import json
import logging
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.optimize import LBFGS
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from scipy.spatial.distance import cosine

from .calculator import get_orb_calculator
from .operations import Operations
from .utils import parse_chemical_formula

# ===================== Constants & Logging =====================

# Generally 0.01–0.1 is common; this is domain-specific. Keep your original default.
SYMPREC = 0.1

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Avoid "No handler found" warnings; user/app can configure logging level/handlers.
    logger.addHandler(logging.NullHandler())


# ===================== Helper utilities =====================


def _angles_to_radians(
    alpha: float, beta: float, gamma: float
) -> Tuple[float, float, float]:
    """Auto-detect degrees vs radians and return radians."""
    angles = [float(alpha), float(beta), float(gamma)]
    if max(angles) <= math.pi + 1e-3:  # Already radians
        return angles[0], angles[1], angles[2]
    return math.radians(alpha), math.radians(beta), math.radians(gamma)


def _cell_from_abc_alpha_beta_gamma(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Construct a 3x3 cell matrix from lattice parameters (angles in degrees or radians)."""
    alpha_rad, beta_rad, gamma_rad = _angles_to_radians(alpha, beta, gamma)
    cell = np.zeros((3, 3), dtype=float)
    cell[0, 0] = a
    cell[1, 0] = b * math.cos(gamma_rad)
    cell[1, 1] = b * math.sin(gamma_rad)
    cell[2, 0] = c * math.cos(beta_rad)
    # guard sin(gamma) to avoid division by ~0
    s_g = math.sin(gamma_rad) if abs(math.sin(gamma_rad)) > 1e-12 else 1e-12
    cell[2, 1] = (
        c * (math.cos(alpha_rad) - math.cos(beta_rad) * math.cos(gamma_rad)) / s_g
    )
    term = (
        1
        - math.cos(alpha_rad) ** 2
        - math.cos(beta_rad) ** 2
        - math.cos(gamma_rad) ** 2
        + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)
    )
    # numerical safety
    term = max(term, 0.0)
    cell[2, 2] = c * math.sqrt(term) / s_g
    return cell


def _structure_from_description_like(
    desc: dict,
) -> Structure:
    """Build a pymatgen Structure from a description/json (space_group, lattice, wyckoff_sites)."""
    space_group = desc["space_group"]
    lattice_params = desc["lattice"]
    wyckoff_sites = desc["wyckoff_sites"]

    a, b, c = (
        float(lattice_params["a"]),
        float(lattice_params["b"]),
        float(lattice_params["c"]),
    )
    alpha, beta, gamma = (
        lattice_params["alpha"],
        lattice_params["beta"],
        lattice_params["gamma"],
    )
    cell = _cell_from_abc_alpha_beta_gamma(a, b, c, alpha, beta, gamma)
    lattice = Lattice(cell)

    # Try to get symmetry operations from number (or fallback to symbol)
    sg_obj: Optional[SpaceGroup] = None
    try:
        sg_obj = SpaceGroup.from_int_number(int(space_group["number"]))
    except Exception:
        try:
            sg_symbol = space_group.get("symbol")
            if sg_symbol:
                sg_obj = SpaceGroup(sg_symbol)
        except Exception:
            sg_obj = None

    symbols: List[str] = []
    positions: List[List[float]] = []

    if sg_obj is not None:
        symm_ops = sg_obj.symmetry_ops
        for site in wyckoff_sites:
            rep = np.array(site["coordinates"], dtype=float)
            # de-duplicate symm-generated positions with rounding
            unique: Dict[Tuple[float, float, float], np.ndarray] = {}
            for op in symm_ops:
                pos = op.operate(rep) % 1.0
                key = tuple(np.round(pos, 6))
                unique[key] = pos
            for pos in unique.values():
                symbols.append(site["element"])
                positions.append([float(v) for v in pos.tolist()])
    else:
        # Fallback: single representative per site
        for site in wyckoff_sites:
            symbols.append(site["element"])
            positions.append([float(x) for x in site["coordinates"]])

    structure = Structure(lattice, symbols, positions)
    # Store provided space-group meta (optional)
    try:
        structure.properties = {"spacegroup": dict(space_group)}
    except Exception:
        pass
    return structure


def _atoms_from_structure(structure: Structure):
    return AseAtomsAdaptor().get_atoms(structure)


def _wrap_to_unit_cell(structure: Structure) -> Structure:
    """Wrap all fractional coordinates to [0, 1) range.

    After relaxation, atoms can drift outside the unit cell. This wraps
    them back to ensure clean visualization and proper symmetry detection.
    """
    new_coords = []
    for site in structure:
        wrapped = site.frac_coords % 1.0
        # Handle numerical precision issues near 1.0
        wrapped = np.where(np.abs(wrapped - 1.0) < 1e-10, 0.0, wrapped)
        new_coords.append(wrapped)

    return Structure(
        structure.lattice,
        [site.specie for site in structure],
        new_coords,
        site_properties=(
            structure.site_properties if structure.site_properties else None
        ),
    )


def _atoms_to_structure_wrapped(atoms) -> Structure:
    """Convert ASE atoms to pymatgen Structure with wrapped coordinates.

    This suppresses the harmless warning about unsupported constraints (like FixSymmetry)
    and ensures coordinates are wrapped to [0, 1).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Only FixAtoms and FixCartesian is supported",
            category=UserWarning,
        )
        structure = AseAtomsAdaptor().get_structure(atoms)
    return _wrap_to_unit_cell(structure)


def _structures_to_atoms_list(structures: List[Structure]):
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_atoms(s) for s in structures]


def get_structure_fingerprint(structure: Structure) -> np.ndarray:
    """Simple fingerprint based on coordination and composition.

    Args:
        structure: pymatgen Structure object

    Returns:
        numpy array with fingerprint features
    """
    # Composition fingerprint
    comp_vector = np.zeros(103)  # for all elements
    for site in structure:
        comp_vector[site.specie.Z - 1] += 1
    comp_vector /= len(structure)

    # Coordination fingerprint
    cnn = CrystalNN()
    coord_list = []
    for i in range(len(structure)):
        try:
            cn = cnn.get_cn(structure, i)
            coord_list.append(cn)
        except:
            coord_list.append(0)

    # Combine features
    features = np.concatenate(
        [
            comp_vector,
            [np.mean(coord_list), np.std(coord_list)],
            [structure.volume / len(structure)],  # Volume per atom
        ]
    )
    return features


# ===================== GGen =====================


class GGen:
    """Core crystal generation functionality using PyXtal and ORB calculator."""

    def __init__(
        self,
        calculator=None,
        random_seed: Optional[int] = None,
        enable_trajectory: bool = True,
    ):
        """Initialize the crystal generator.

        Args:
            calculator: ASE calculator instance. If None, uses ORB calculator.
            random_seed: Optional random seed for reproducible operations.
            enable_trajectory: Whether to enable trajectory tracking for mutations and relaxations.
        """
        self.calculator = calculator or get_orb_calculator()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.crystal_ops = Operations(random_seed=random_seed)
        self._current_structure: Optional[Structure] = None
        self._current_pyxtal: Optional[pyxtal] = None

        # Trajectory tracking using pymatgen structures
        self.enable_trajectory = enable_trajectory
        self._trajectory_structures: List[Structure] = []
        self._trajectory_metadata: List[Dict[str, Any]] = []
        self._trajectory_info: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "random_seed": random_seed,
            "total_frames": 0,
        }

    # -------------------- Space-group utilities --------------------

    def get_compatible_space_groups(
        self, elements: List[str], counts: List[int]
    ) -> List[Dict[str, Any]]:
        """Get space groups compatible with a given stoichiometry.

        Note: Compatibility is based on the multiplicities / Wyckoff partitioning of `counts` only.

        Args:
            elements: List of element symbols (not used in the compatibility check).
            counts: List of stoichiometric coefficients.

        Returns:
            List of dictionaries containing compatible space-group info.
        """
        compatible_groups: List[Dict[str, Any]] = []

        for sg_number in range(1, 231):  # International space groups 1–230
            try:
                group = Group(sg_number, dim=3)
                is_compatible, _ = group.check_compatible(counts)
                if is_compatible:
                    wyckoff_positions = group.Wyckoff_positions
                    compatible_groups.append(
                        {
                            "number": sg_number,
                            "symbol": group.symbol,
                            "wyckoff_positions": len(wyckoff_positions),
                            "chiral": group.chiral,
                            "polar": group.polar,
                            "inversion": group.inversion,
                            "lattice_type": getattr(group, "lattice_type", None),
                        }
                    )
            except Exception:
                continue

        return compatible_groups

    def select_random_space_group_with_symmetry_preference(
        self,
        compatible_groups: List[Dict[str, Any]],
        random_seed: Optional[int] = None,
        symmetry_bias: float = 0.0,
    ) -> int:
        """Select a random space group from compatible groups with configurable symmetry preference.

        Args:
            compatible_groups: List of compatible space group dictionaries.
            random_seed: Optional random seed for reproducibility.
            symmetry_bias: Controls preference for higher-symmetry crystal systems.
                0.0 = uniform distribution (equal weight for orthorhombic and above,
                      reduced weight for triclinic/monoclinic)
                1.0 = strong preference for cubic/hexagonal systems
                Default: 0.0

        Returns:
            Selected space group number.
        """
        if not compatible_groups:
            raise ValueError("No compatible space groups provided")

        rng = np.random.default_rng(
            self.random_seed if random_seed is None else random_seed
        )

        # Clamp symmetry_bias to [0, 1]
        symmetry_bias = max(0.0, min(1.0, symmetry_bias))

        # Base weights: uniform above monoclinic, reduced for triclinic/monoclinic
        uniform_weights = {
            (1, 2): 0.5,  # triclinic - reduced
            (3, 15): 0.5,  # monoclinic - reduced
            (16, 74): 1.0,  # orthorhombic
            (75, 142): 1.0,  # tetragonal
            (143, 167): 1.0,  # trigonal
            (168, 194): 1.0,  # hexagonal
            (195, 230): 1.0,  # cubic
        }

        # Biased weights: strong preference for higher symmetry
        biased_weights = {
            (1, 2): 1.0,  # triclinic
            (3, 15): 2.0,  # monoclinic
            (16, 74): 3.0,  # orthorhombic
            (75, 142): 4.0,  # tetragonal
            (143, 167): 4.5,  # trigonal
            (168, 194): 5.0,  # hexagonal
            (195, 230): 6.0,  # cubic
        }

        def _sys_weight(sg: int) -> float:
            for a, b in uniform_weights.keys():
                if a <= sg <= b:
                    uniform_w = uniform_weights[(a, b)]
                    biased_w = biased_weights[(a, b)]
                    # Interpolate between uniform and biased based on symmetry_bias
                    return uniform_w + symmetry_bias * (biased_w - uniform_w)
            return 1.0

        weights = []
        for g in compatible_groups:
            sg = g["number"]
            base = _sys_weight(sg)
            # Small bonuses for structural features (reduced from original)
            wyckoff_bonus = (
                min(g.get("wyckoff_positions", 1) / 20.0, 0.5) * symmetry_bias
            )
            chiral_bonus = 0.05 * symmetry_bias if g.get("chiral", False) else 0.0
            inversion_bonus = 0.1 * symmetry_bias if g.get("inversion", False) else 0.0
            # Remove sgnum_bonus entirely - it was adding extra cubic bias
            weights.append(base + wyckoff_bonus + chiral_bonus + inversion_bonus)

        probs = np.array(weights, dtype=float)
        probs /= probs.sum()

        idx = rng.choice(len(compatible_groups), p=probs)
        return int(compatible_groups[idx]["number"])

    # -------------------- Crystal quality & scoring utilities --------------------

    def _get_crystal_system_name(self, sg_number: int) -> str:
        """Get crystal system name from space group number."""
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

    def _is_structurally_valid(
        self,
        structure: Structure,
        min_distance: float = 1.0,
        min_volume_per_atom: float = 2.0,
        max_volume_per_atom: float = 100.0,
    ) -> Tuple[bool, str]:
        """Quick validity check before expensive energy evaluation.

        Args:
            structure: Structure to validate
            min_distance: Minimum allowed interatomic distance (Å)
            min_volume_per_atom: Minimum volume per atom (Å³)
            max_volume_per_atom: Maximum volume per atom (Å³)

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if len(structure) == 0:
            return False, "empty structure"

        # Check volume per atom
        vol_per_atom = structure.volume / len(structure)
        if vol_per_atom < min_volume_per_atom:
            return False, f"volume per atom too small: {vol_per_atom:.2f} Å³"
        if vol_per_atom > max_volume_per_atom:
            return False, f"volume per atom too large: {vol_per_atom:.2f} Å³"

        # Check for overlapping atoms (computationally cheap for small structures)
        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                dist = structure.get_distance(i, j)
                if dist < min_distance:
                    return False, f"atoms {i} and {j} too close: {dist:.3f} Å"

        return True, "valid"

    def _score_crystal(
        self,
        crystal: pyxtal,
        energy: float,
        symmetry_weight: float = 0.1,
        wyckoff_weight: float = 0.01,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a crystal considering both energy and symmetry.

        Lower scores are better.

        Args:
            crystal: PyXtal crystal object
            energy: Calculated energy (eV)
            symmetry_weight: Weight for symmetry bonus (higher SG = lower score)
            wyckoff_weight: Penalty per unique Wyckoff site

        Returns:
            Tuple of (total_score, score_breakdown)
        """
        # Energy component (lower is better)
        energy_score = energy

        # Symmetry bonus: higher SG number generally means higher symmetry
        sg_number = crystal.group.number
        symmetry_bonus = -sg_number * symmetry_weight

        # Wyckoff efficiency: fewer unique sites = more symmetric/simpler
        num_wyckoff_sites = len(crystal.atom_sites)
        wyckoff_penalty = num_wyckoff_sites * wyckoff_weight

        total_score = energy_score + symmetry_bonus + wyckoff_penalty

        breakdown = {
            "energy": energy_score,
            "symmetry_bonus": symmetry_bonus,
            "wyckoff_penalty": wyckoff_penalty,
            "total": total_score,
            "space_group": sg_number,
            "num_wyckoff_sites": num_wyckoff_sites,
        }

        return total_score, breakdown

    def _refine_to_higher_symmetry(
        self,
        structure: Structure,
        tolerance_steps: List[float] = None,
    ) -> Tuple[Structure, int, str]:
        """Attempt to find higher-symmetry description of structure.

        Tries increasingly loose tolerances to detect hidden symmetry.

        Args:
            structure: Structure to refine
            tolerance_steps: List of symprec values to try (default: [0.01, 0.05, 0.1, 0.2])

        Returns:
            Tuple of (refined_structure, detected_sg_number, detected_sg_symbol)
        """
        if tolerance_steps is None:
            tolerance_steps = [0.01, 0.05, 0.1, 0.2]

        best_structure = structure
        best_sg_number = 1
        best_sg_symbol = "P1"

        for symprec in tolerance_steps:
            try:
                spg = SpacegroupAnalyzer(structure, symprec=symprec)
                sg_number = spg.get_space_group_number()

                if sg_number > best_sg_number:
                    refined = spg.get_refined_structure()
                    best_structure = refined
                    best_sg_number = sg_number
                    best_sg_symbol = spg.get_space_group_symbol()
                    logger.debug(
                        "Found higher symmetry at symprec=%.3f: %s (#%d)",
                        symprec,
                        best_sg_symbol,
                        best_sg_number,
                    )
            except Exception as e:
                logger.debug(
                    "Symmetry refinement failed at symprec=%.3f: %s", symprec, e
                )
                continue

        return best_structure, best_sg_number, best_sg_symbol

    def _select_top_space_groups(
        self,
        compatible_groups: List[Dict[str, Any]],
        top_k: int = 5,
        symmetry_bias: float = 0.0,
    ) -> List[int]:
        """Select top K space groups with configurable symmetry preference.

        Args:
            compatible_groups: List of compatible space group dicts
            top_k: Number of space groups to return
            symmetry_bias: Controls selection strategy.
                0.0 = sample uniformly across crystal systems (above monoclinic)
                1.0 = prioritize highest-symmetry space groups
                Default: 0.0

        Returns:
            List of space group numbers
        """
        if not compatible_groups:
            return []

        symmetry_bias = max(0.0, min(1.0, symmetry_bias))

        if symmetry_bias >= 0.8:
            # High bias: just take highest space group numbers
            sorted_groups = sorted(
                compatible_groups, key=lambda x: x["number"], reverse=True
            )
            return [g["number"] for g in sorted_groups[:top_k]]

        # Lower bias: sample to get diversity across crystal systems
        # Group by crystal system
        crystal_systems = {
            "triclinic": [],
            "monoclinic": [],
            "orthorhombic": [],
            "tetragonal": [],
            "trigonal": [],
            "hexagonal": [],
            "cubic": [],
        }

        for g in compatible_groups:
            sg = g["number"]
            system = self._get_crystal_system_name(sg)
            if system in crystal_systems:
                crystal_systems[system].append(g)

        # Build selection with representation from each non-empty system
        # Prioritize systems above monoclinic
        priority_systems = [
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        ]
        low_priority_systems = ["triclinic", "monoclinic"]

        selected = []
        rng = np.random.default_rng(self.random_seed)

        # First pass: one from each priority system that has compatible groups
        for system in priority_systems:
            if crystal_systems[system] and len(selected) < top_k:
                # Pick one randomly from this system
                group = rng.choice(crystal_systems[system])
                selected.append(group["number"])

        # Second pass: fill remaining slots, allowing duplicates from systems
        all_priority = []
        for system in priority_systems:
            all_priority.extend(crystal_systems[system])

        while len(selected) < top_k and all_priority:
            group = rng.choice(all_priority)
            if group["number"] not in selected:
                selected.append(group["number"])
            # Remove to avoid infinite loop if we run out
            all_priority = [g for g in all_priority if g["number"] not in selected]

        # If still not enough, add from low priority systems
        if len(selected) < top_k:
            for system in low_priority_systems:
                for g in crystal_systems[system]:
                    if g["number"] not in selected and len(selected) < top_k:
                        selected.append(g["number"])

        return selected

    # -------------------- Trajectory management --------------------

    def _add_trajectory_frame(
        self,
        structure: Structure,
        frame_type: str,
        operation: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a frame to the trajectory using pymatgen structures."""
        if not self.enable_trajectory:
            return

        if len(structure) == 0:
            logger.warning("Skipping empty structure in trajectory frame.")
            return

        # Always get atoms first so we can collect cell/composition even if energy fails
        atoms = _atoms_from_structure(structure)

        # Calculate energy if possible
        energy = None
        try:
            atoms.calc = self.calculator
            energy = atoms.get_potential_energy()
        except Exception as e:
            logger.debug("Energy evaluation failed in trajectory frame: %s", e)

        # Store the structure snapshot
        self._trajectory_structures.append(structure.copy())

        # Space-group analysis
        sg_meta: Optional[Dict[str, Any]] = None
        try:
            spg_analyzer = SpacegroupAnalyzer(structure, symprec=SYMPREC)
            sg_meta = {
                "number": spg_analyzer.get_space_group_number(),
                "symbol": spg_analyzer.get_space_group_symbol(),
                "crystal_system": spg_analyzer.get_crystal_system(),
            }
        except Exception as e:
            logger.debug("Space-group analysis failed for trajectory frame: %s", e)
            sg_meta = None

        cell = atoms.cell
        frame_metadata = {
            "frame_index": len(self._trajectory_structures) - 1,
            "timestamp": datetime.now().isoformat(),
            "frame_type": frame_type,
            "operation": operation,
            "parameters": parameters or {},
            "energy": energy,
            "composition": atoms.get_chemical_formula(),
            "num_sites": len(atoms),
            "space_group": sg_meta,
            "lattice": {
                "a": float(cell.lengths()[0]),
                "b": float(cell.lengths()[1]),
                "c": float(cell.lengths()[2]),
                "alpha": float(cell.angles()[0]),
                "beta": float(cell.angles()[1]),
                "gamma": float(cell.angles()[2]),
                "volume": float(cell.volume),
            },
            "metadata": metadata or {},
        }

        self._trajectory_metadata.append(frame_metadata)
        self._trajectory_info["total_frames"] = len(self._trajectory_structures)

    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Return trajectory frames as list[dict] with attached Structure objects."""
        return [
            {"structure": s, **m}
            for s, m in zip(self._trajectory_structures, self._trajectory_metadata)
        ]

    def get_trajectory_structures(self) -> List[Structure]:
        return [s.copy() for s in self._trajectory_structures]

    def get_trajectory_metadata(self) -> Dict[str, Any]:
        return dict(self._trajectory_info)

    def clear_trajectory(self) -> None:
        self._trajectory_structures.clear()
        self._trajectory_metadata.clear()
        self._trajectory_info["total_frames"] = 0

    def export_trajectory(self, filename: Optional[str] = None) -> str:
        """Export trajectory as a standard ASE .traj file for use in other tools.

        The .traj format is widely supported by ASE and other computational chemistry tools.
        Each frame includes atomic positions, cell vectors, and metadata.

        Args:
            filename: Output filename. If None, auto-generates with timestamp.

        Returns:
            The filename of the created .traj file.

        Raises:
            ValueError: If no trajectory data is available to export.
        """
        from ase.io import write

        if not self._trajectory_structures:
            raise ValueError("No trajectory data available to export")

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crystal_trajectory_{ts}.traj"

        # Convert structures to ASE atoms objects
        atoms_list = _structures_to_atoms_list(self._trajectory_structures)

        # Add metadata to each atoms object
        for i, (atoms, meta) in enumerate(zip(atoms_list, self._trajectory_metadata)):
            # Add frame metadata as info
            atoms.info.update(
                {
                    "frame_index": meta.get("frame_index", i),
                    "frame_type": meta.get("frame_type", "unknown"),
                    "operation": meta.get("operation", "unknown"),
                    "timestamp": meta.get("timestamp", ""),
                    "energy": meta.get("energy"),
                    "composition": meta.get("composition", ""),
                    "num_sites": meta.get("num_sites", len(atoms)),
                }
            )

            # Add space group info if available
            if meta.get("space_group"):
                atoms.info["space_group"] = meta["space_group"]

            # Add lattice parameters if available
            if meta.get("lattice"):
                atoms.info["lattice"] = meta["lattice"]

            # Add any additional metadata
            if meta.get("metadata"):
                atoms.info["metadata"] = meta["metadata"]

        # Write to .traj file using ASE's Trajectory class
        from ase.io.trajectory import Trajectory

        with Trajectory(filename, "w") as traj:
            for atoms in atoms_list:
                traj.write(atoms)

        return filename

    # -------------------- Geometry optimization --------------------

    def optimize_geometry(
        self,
        max_steps: int = 400,
        fmax: float = 0.01,
        relax_cell: bool = True,
        trajectory_interval: int = 5,
        preserve_symmetry: bool = False,
        symmetry_symprec: float = 0.01,
    ) -> Tuple[Structure, float, int]:
        """Optimize geometry using ASE LBFGS with optional variable cell.

        When preserve_symmetry=True, uses a two-stage approach:
        1. Symmetry-constrained relaxation with FixSymmetry constraint
        2. Unconstrained fine-tuning to allow true symmetry-breaking instabilities

        Args:
            max_steps: Maximum optimization steps (used for both stages if preserve_symmetry).
            fmax: Force convergence criterion (eV/Å).
            relax_cell: Whether to allow cell shape/volume changes.
            trajectory_interval: Steps between trajectory snapshots.
            preserve_symmetry: If True, use two-stage symmetry-preserving relaxation.
                Stage 1 applies FixSymmetry constraint to keep atoms on Wyckoff positions.
                Stage 2 does unconstrained fine-tuning with conservative settings.
            symmetry_symprec: Tolerance for symmetry detection when preserve_symmetry
                is True. Smaller values are stricter. Default: 0.01 Å.

        Returns:
            (optimized_structure, final_energy_eV, total_steps)
        """
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )

        if self.enable_trajectory:
            self._add_trajectory_frame(
                structure=self._current_structure,
                frame_type="optimization_start",
                operation="optimize_geometry",
                parameters={
                    "max_steps": max_steps,
                    "fmax": fmax,
                    "relax_cell": relax_cell,
                    "preserve_symmetry": preserve_symmetry,
                },
                metadata={"step": 0},
            )

        atoms = _atoms_from_structure(self._current_structure)
        atoms.calc = self.calculator

        # Apply symmetry constraint if requested
        if preserve_symmetry:
            try:
                sym_constraint = FixSymmetry(
                    atoms,
                    symprec=symmetry_symprec,
                    adjust_positions=True,
                    adjust_cell=True,
                )
                atoms.set_constraint(sym_constraint)
                logger.info(
                    "Symmetry constraint applied (symprec=%.4f) - forces will be projected "
                    "onto symmetry-allowed subspace",
                    symmetry_symprec,
                )
            except Exception as e:
                logger.warning(
                    "Failed to apply symmetry constraint: %s. Proceeding without.", e
                )

        dyn_target = FrechetCellFilter(atoms) if relax_cell else atoms

        if self.enable_trajectory and trajectory_interval > 0:

            class TrajectoryLBFGS(LBFGS):
                def __init__(self, atoms, ggen: GGen, interval: int, **kwargs):
                    super().__init__(atoms, **kwargs)
                    self.ggen = ggen
                    self.interval = interval
                    self.step_count = 0

                def step(self, f=None):
                    result = super().step(f)
                    self.step_count += 1
                    if self.step_count % self.interval == 0:
                        # FrechetCellFilter has .atoms
                        a = (
                            self.atoms.atoms
                            if hasattr(self.atoms, "atoms")
                            else self.atoms
                        )
                        structure = _atoms_to_structure_wrapped(a)
                        self.ggen._add_trajectory_frame(
                            structure=structure,
                            frame_type="optimization_step",
                            operation="optimize_geometry",
                            metadata={"step": self.step_count},
                        )
                    return result

            optimizer = TrajectoryLBFGS(
                dyn_target, self, trajectory_interval, maxstep=0.2, logfile=None
            )
        else:
            optimizer = LBFGS(dyn_target, maxstep=0.2, logfile=None)

        try:
            optimizer.run(fmax=fmax, steps=max_steps)
            final_energy = atoms.get_potential_energy()
            num_steps = optimizer.get_number_of_steps()

            if self.enable_trajectory:
                final_struct = _atoms_to_structure_wrapped(atoms)
                self._add_trajectory_frame(
                    structure=final_struct,
                    frame_type=(
                        "optimization_end" if not preserve_symmetry else "stage1_end"
                    ),
                    operation="optimize_geometry",
                    parameters={
                        "max_steps": max_steps,
                        "fmax": fmax,
                        "relax_cell": relax_cell,
                        "steps_completed": num_steps,
                    },
                    metadata={"step": num_steps, "converged": True},
                )

            optimized_structure = _atoms_to_structure_wrapped(atoms)
            self.set_structure(optimized_structure, add_to_trajectory=False)

            # Stage 2: unconstrained fine-tuning after symmetry-constrained relaxation
            if preserve_symmetry:
                logger.info(
                    "Stage 1 complete: %d steps, E=%.4f eV", num_steps, final_energy
                )

                # Get symmetry after stage 1 for comparison
                spg1 = SpacegroupAnalyzer(optimized_structure, symprec=SYMPREC)
                sg1 = spg1.get_space_group_number()
                logger.info(
                    "Stage 1 symmetry: SG=%d (%s)", sg1, spg1.get_space_group_symbol()
                )

                logger.info("Stage 2: Unconstrained fine-tuning...")

                # Fresh atoms without symmetry constraint
                atoms2 = _atoms_from_structure(optimized_structure)
                atoms2.calc = self.calculator
                dyn_target2 = FrechetCellFilter(atoms2) if relax_cell else atoms2

                # Conservative optimizer settings to avoid escaping high-symmetry basin
                optimizer2 = LBFGS(dyn_target2, maxstep=0.05, logfile=None)

                try:
                    optimizer2.run(fmax=fmax, steps=max_steps)
                    final_energy = atoms2.get_potential_energy()
                    steps2 = optimizer2.get_number_of_steps()
                    num_steps += steps2

                    optimized_structure = _atoms_to_structure_wrapped(atoms2)
                    self.set_structure(optimized_structure, add_to_trajectory=False)

                    # Check if symmetry was lost
                    spg2 = SpacegroupAnalyzer(optimized_structure, symprec=SYMPREC)
                    sg2 = spg2.get_space_group_number()

                    if sg2 < sg1:
                        logger.info(
                            "Stage 2: Symmetry reduced %d -> %d (true instability), "
                            "%d steps, E=%.4f eV",
                            sg1,
                            sg2,
                            steps2,
                            final_energy,
                        )
                    else:
                        logger.info(
                            "Stage 2 complete: %d steps, E=%.4f eV, symmetry preserved SG=%d",
                            steps2,
                            final_energy,
                            sg2,
                        )

                    if self.enable_trajectory:
                        self._add_trajectory_frame(
                            structure=optimized_structure,
                            frame_type="optimization_end",
                            operation="optimize_geometry",
                            parameters={
                                "stage2_steps": steps2,
                                "stage2_fmax": fmax,
                            },
                            metadata={"stage": 2, "converged": True},
                        )

                except Exception as e:
                    logger.warning("Stage 2 failed: %s. Using stage 1 result.", e)

            return optimized_structure, float(final_energy), int(num_steps)

        except Exception as e:
            logger.error("Geometry optimization failed: %s", e)
            try:
                original_energy = float(atoms.get_potential_energy())
            except Exception:
                original_energy = float("nan")

            if self.enable_trajectory:
                fallback_struct = _atoms_to_structure_wrapped(atoms)
                self._add_trajectory_frame(
                    structure=fallback_struct,
                    frame_type="optimization_failed",
                    operation="optimize_geometry",
                    parameters={
                        "max_steps": max_steps,
                        "fmax": fmax,
                        "relax_cell": relax_cell,
                    },
                    metadata={"error": str(e), "converged": False},
                )
            return self._current_structure, original_energy, 0

    # -------------------- Crystal generation --------------------

    def generate_crystal(
        self,
        formula: str,
        space_group: Optional[int] = None,
        num_trials: int = 10,
        optimize_geometry: bool = False,
        # Enhanced generation parameters
        multi_spacegroup: bool = True,
        top_k_spacegroups: int = 5,
        symmetry_weight: float = 0.01,
        symmetry_bias: float = 0.0,
        crystal_systems: Optional[List[str]] = None,
        refine_symmetry: bool = True,
        min_distance_filter: float = 1.0,
        volume_bounds: Tuple[float, float] = (2.0, 100.0),
        preserve_symmetry: bool = False,
        # Iterative refinement parameters
        max_iterations: int = 1,
        convergence_threshold: float = 0.01,
        # Optimization parameters
        optimization_max_steps: int = 400,
        optimization_fmax: float = 0.01,
        trajectory_interval: int = 5,
    ) -> Dict[str, Any]:
        """Generate a crystal structure using PyXtal with enhanced stability and symmetry.

        This method implements an improved generation workflow:
        1. Multi-space-group exploration (tries multiple compatible space groups)
        2. Pre-filtering of invalid structures before expensive energy evaluation
        3. Combined scoring considering both energy and symmetry
        4. Optional post-relaxation symmetry refinement
        5. Optional iterative refinement (when max_iterations > 1)

        When max_iterations > 1, uses iterative refinement:
        - Generate and relax the best structure
        - Detect actual symmetry after relaxation
        - Regenerate in detected symmetry space group
        - Repeat until convergence or max iterations

        Args:
            formula: Chemical formula (e.g., "NaCl", "TiO2")
            space_group: Specific space group number to use. If None, auto-selects.
            num_trials: Number of random structures to generate per space group.
            optimize_geometry: Whether to relax the structure after generation.
            multi_spacegroup: If True and space_group is None, tries multiple space groups.
            top_k_spacegroups: Number of top space groups to try when multi_spacegroup=True.
            symmetry_weight: Weight for symmetry in combined scoring (higher = prefer symmetry).
            symmetry_bias: Controls preference for higher-symmetry crystal systems when
                auto-selecting space groups. 0.0 = uniform distribution across orthorhombic
                and above (reduced weight for triclinic/monoclinic). 1.0 = strong preference
                for cubic/hexagonal. Default: 0.0
            crystal_systems: Optional list of crystal systems to restrict generation to.
                Valid values: "triclinic", "monoclinic", "orthorhombic", "tetragonal",
                "trigonal", "hexagonal", "cubic". If None, all systems are considered.
                Example: ["tetragonal", "hexagonal"] to only generate in those systems.
            refine_symmetry: Whether to attempt symmetry refinement after relaxation.
            min_distance_filter: Minimum interatomic distance for pre-filtering (Å).
            volume_bounds: (min, max) volume per atom bounds for pre-filtering (Å³).
            preserve_symmetry: If True, use symmetry-constrained relaxation to keep atoms
                on Wyckoff positions during optimization. This uses a two-stage approach:
                (1) symmetry-constrained relaxation, then (2) brief unconstrained fine-tuning.
                This preserves high symmetry when energetically reasonable. Default: False.
            max_iterations: Maximum generate-relax-regenerate cycles. When > 1, enables
                iterative refinement where the structure is regenerated in the detected
                symmetry space group. Default: 1 (single-shot).
            convergence_threshold: Energy change threshold (eV) for convergence in iterative
                mode. Stops when improvement is less than this. Default: 0.01.
            optimization_max_steps: Max steps for geometry optimization. Default: 400.
            optimization_fmax: Force convergence criterion (eV/Å). Default: 0.01.
            trajectory_interval: Steps between trajectory frame snapshots during
                optimization. Set to 0 to disable intermediate frames. Default: 5.

        Returns:
            Dictionary with structure metadata, CIF content, and generation statistics.
            When max_iterations > 1, also includes iteration_history and convergence info.
        """
        # Handle iterative refinement mode
        if max_iterations > 1:
            return self._generate_crystal_iterative(
                formula=formula,
                max_iterations=max_iterations,
                num_trials=num_trials,
                convergence_threshold=convergence_threshold,
                top_k_spacegroups=top_k_spacegroups,
                symmetry_weight=symmetry_weight,
                symmetry_bias=symmetry_bias,
                crystal_systems=crystal_systems,
                min_distance_filter=min_distance_filter,
                volume_bounds=volume_bounds,
                preserve_symmetry=preserve_symmetry,
                optimization_max_steps=optimization_max_steps,
                optimization_fmax=optimization_fmax,
                trajectory_interval=trajectory_interval,
            )
        if num_trials < 1:
            raise ValueError("num_trials must be >= 1")
        num_trials = int(min(num_trials, 100))

        elements, counts = parse_chemical_formula(formula)
        logger.info(
            "Starting crystal generation for %s (elements=%s, counts=%s)",
            formula,
            elements,
            counts,
        )

        # Get compatible space groups
        compatible = self.get_compatible_space_groups(elements, counts)
        if not compatible:
            raise ValueError(
                f"No compatible space groups found for composition {formula}"
            )
        logger.info("Found %d compatible space groups", len(compatible))

        # Filter by crystal system if specified
        if crystal_systems is not None:
            valid_systems = {
                "triclinic",
                "monoclinic",
                "orthorhombic",
                "tetragonal",
                "trigonal",
                "hexagonal",
                "cubic",
            }
            # Normalize to lowercase
            crystal_systems_lower = [s.lower() for s in crystal_systems]
            invalid = set(crystal_systems_lower) - valid_systems
            if invalid:
                raise ValueError(
                    f"Invalid crystal system(s): {invalid}. "
                    f"Valid options: {sorted(valid_systems)}"
                )

            # Filter compatible groups to only those in specified systems
            compatible = [
                g
                for g in compatible
                if self._get_crystal_system_name(g["number"]) in crystal_systems_lower
            ]
            if not compatible:
                raise ValueError(
                    f"No compatible space groups found for composition {formula} "
                    f"in crystal system(s): {crystal_systems}"
                )
            logger.info(
                "Filtered to %d space groups in systems: %s",
                len(compatible),
                crystal_systems,
            )

        # Determine which space groups to try
        target_space_groups: List[int] = []
        was_randomly_selected = False

        if space_group is not None:
            # User specified a space group - validate it
            try:
                g = Group(int(space_group), dim=3)
                ok, _msg = g.check_compatible(counts)
                if not ok:
                    raise ValueError(
                        f"Composition {counts} not compatible with space group {space_group}. "
                        f"Compatible space groups: {[x['number'] for x in compatible]}"
                    )
                target_space_groups = [space_group]
                logger.info("Using user-specified space group: %d", space_group)
            except Exception as e:
                raise ValueError(
                    f"Space group validation failed: {e}. "
                    f"Compatible space groups: {[x['number'] for x in compatible]}"
                )
        elif multi_spacegroup:
            # Try multiple space groups with configurable symmetry preference
            target_space_groups = self._select_top_space_groups(
                compatible, top_k=top_k_spacegroups, symmetry_bias=symmetry_bias
            )
            was_randomly_selected = True
            logger.info(
                "Multi-spacegroup mode (symmetry_bias=%.2f): trying %d space groups: %s",
                symmetry_bias,
                len(target_space_groups),
                target_space_groups,
            )
        else:
            # Single auto-selected space group
            selected = self.select_random_space_group_with_symmetry_preference(
                compatible, random_seed=self.random_seed, symmetry_bias=symmetry_bias
            )
            target_space_groups = [selected]
            was_randomly_selected = True
            logger.info(
                "Auto-selected space group (symmetry_bias=%.2f): %d",
                symmetry_bias,
                selected,
            )

        # Generate crystals across all target space groups
        all_candidates: List[Tuple[pyxtal, float, Dict[str, float], int]] = []
        generation_stats = {
            "total_attempts": 0,
            "valid_pyxtal": 0,
            "passed_prefilter": 0,
            "energy_evaluated": 0,
            "by_spacegroup": {},
        }

        trials_per_sg = max(1, num_trials // len(target_space_groups))

        for sg_number in target_space_groups:
            sg_stats = {"attempts": 0, "valid": 0, "passed_filter": 0}
            logger.info(
                "Generating %d trials in space group %d (%s)",
                trials_per_sg,
                sg_number,
                self._get_crystal_system_name(sg_number),
            )

            for trial_idx in range(trials_per_sg):
                generation_stats["total_attempts"] += 1
                sg_stats["attempts"] += 1

                # Generate random crystal
                c = pyxtal()
                try:
                    c.from_random(
                        dim=3,
                        group=sg_number,
                        species=elements,
                        numIons=counts,
                        seed=self.random_seed,
                    )
                except TypeError:
                    # Older pyxtal versions may not accept seed
                    c.from_random(
                        dim=3, group=sg_number, species=elements, numIons=counts
                    )

                if not c.valid:
                    logger.debug(
                        "SG %d trial %d: PyXtal generation failed (invalid)",
                        sg_number,
                        trial_idx + 1,
                    )
                    continue

                generation_stats["valid_pyxtal"] += 1
                sg_stats["valid"] += 1

                # Pre-filter: check structural validity before energy evaluation
                structure = c.to_pymatgen()
                is_valid, reason = self._is_structurally_valid(
                    structure,
                    min_distance=min_distance_filter,
                    min_volume_per_atom=volume_bounds[0],
                    max_volume_per_atom=volume_bounds[1],
                )

                if not is_valid:
                    logger.debug(
                        "SG %d trial %d: Pre-filter rejected (%s)",
                        sg_number,
                        trial_idx + 1,
                        reason,
                    )
                    continue

                generation_stats["passed_prefilter"] += 1
                sg_stats["passed_filter"] += 1

                # Energy evaluation
                try:
                    atoms = c.to_ase()
                    atoms.calc = self.calculator
                    energy = float(atoms.get_potential_energy())
                    generation_stats["energy_evaluated"] += 1

                    # Combined scoring
                    score, breakdown = self._score_crystal(
                        c, energy, symmetry_weight=symmetry_weight
                    )

                    all_candidates.append((c, score, breakdown, sg_number))
                    logger.debug(
                        "SG %d trial %d: energy=%.4f eV, score=%.4f (SG bonus=%.4f)",
                        sg_number,
                        trial_idx + 1,
                        energy,
                        score,
                        breakdown["symmetry_bonus"],
                    )

                except Exception as e:
                    logger.warning(
                        "SG %d trial %d: Energy evaluation failed: %s",
                        sg_number,
                        trial_idx + 1,
                        e,
                    )
                    continue

            generation_stats["by_spacegroup"][sg_number] = sg_stats
            logger.info(
                "SG %d complete: %d/%d valid, %d passed pre-filter",
                sg_number,
                sg_stats["valid"],
                sg_stats["attempts"],
                sg_stats["passed_filter"],
            )

        # Select best candidate
        if not all_candidates:
            raise ValueError(
                "Failed to generate valid crystal structure. "
                f"Stats: {generation_stats}. "
                f"Compatible space groups: {[x['number'] for x in compatible]}"
            )

        # Sort by combined score (lower is better)
        all_candidates.sort(key=lambda x: x[1])
        best_crystal, best_score, best_breakdown, best_sg = all_candidates[0]
        structure = best_crystal.to_pymatgen()
        final_energy = best_breakdown["energy"]
        optimization_steps = 0

        logger.info(
            "Selected best candidate: SG %d, energy=%.4f eV, score=%.4f",
            best_sg,
            final_energy,
            best_score,
        )
        logger.info(
            "Generation stats: %d attempts → %d valid → %d passed filter → %d evaluated",
            generation_stats["total_attempts"],
            generation_stats["valid_pyxtal"],
            generation_stats["passed_prefilter"],
            generation_stats["energy_evaluated"],
        )

        # Geometry optimization
        if optimize_geometry:
            self.set_structure(structure, add_to_trajectory=True)
            logger.info(
                "Starting geometry optimization%s...",
                " (symmetry-preserving)" if preserve_symmetry else "",
            )
            structure, final_energy, optimization_steps = self.optimize_geometry(
                max_steps=optimization_max_steps,
                fmax=optimization_fmax,
                relax_cell=True,
                trajectory_interval=trajectory_interval,
                preserve_symmetry=preserve_symmetry,
            )
            logger.info(
                "Optimization complete: %d steps, final energy=%.4f eV",
                optimization_steps,
                final_energy,
            )

            # Symmetry refinement after relaxation (only if not using preserve_symmetry)
            if refine_symmetry and not preserve_symmetry:
                logger.info("Attempting post-relaxation symmetry refinement...")
                refined_struct, refined_sg, refined_symbol = (
                    self._refine_to_higher_symmetry(structure)
                )
                if refined_sg > 1:
                    original_atoms = len(structure)
                    refined_atoms = len(refined_struct)

                    # Symmetry refinement moves atoms to ideal positions, which may
                    # introduce strain. Do a quick re-relaxation to ensure we're at
                    # a true local minimum with correct energy.
                    logger.info(
                        "Re-relaxing after symmetry refinement (%d -> %d atoms)...",
                        original_atoms,
                        refined_atoms,
                    )
                    self.set_structure(refined_struct, add_to_trajectory=False)
                    refined_struct, final_energy, rerelax_steps = (
                        self.optimize_geometry(
                            max_steps=min(
                                100, optimization_max_steps
                            ),  # Quick re-relax
                            fmax=optimization_fmax,
                            relax_cell=True,
                            trajectory_interval=trajectory_interval,
                        )
                    )
                    optimization_steps += rerelax_steps
                    logger.info(
                        "Post-refinement relaxation: %d steps, energy=%.4f eV",
                        rerelax_steps,
                        final_energy,
                    )

                    structure = refined_struct
                    logger.info(
                        "Symmetry refinement successful: %s (#%d)",
                        refined_symbol,
                        refined_sg,
                    )
                else:
                    logger.info("No higher symmetry detected after relaxation")

        # Final structure and analysis
        self.set_structure(structure, add_to_trajectory=not optimize_geometry)
        spg = SpacegroupAnalyzer(structure, symprec=SYMPREC)
        final_sg_num = spg.get_space_group_number()
        final_sg_sym = spg.get_space_group_symbol()

        requested_space_group_symbol = best_crystal.group.symbol
        filename = f"{formula}_{final_sg_sym.replace('/', '-')}.cif"
        name = f"{formula} ({final_sg_sym})"

        optimization_text = ""
        if optimize_geometry:
            optimization_text = f", optimized: {optimization_steps} steps, cell relaxed"
            if preserve_symmetry:
                optimization_text += ", symmetry-constrained"
            elif refine_symmetry:
                optimization_text += ", symmetry refined"

        # Build description
        if multi_spacegroup and was_randomly_selected:
            description = (
                f"{formula} (best of {len(target_space_groups)} space groups, "
                f"final: {final_sg_sym} #{final_sg_num}{optimization_text})"
            )
        elif was_randomly_selected:
            description = (
                f"{formula} (auto-selected SG: {requested_space_group_symbol} #{best_sg}, "
                f"calculated: {final_sg_sym} #{final_sg_num}{optimization_text})"
            )
        else:
            description = (
                f"{formula} (requested SG: {requested_space_group_symbol} #{space_group}, "
                f"calculated: {final_sg_sym} #{final_sg_num}{optimization_text})"
            )

        # Produce CIF content
        cif_text = str(CifWriter(structure, symprec=SYMPREC))
        cif64 = base64.b64encode(cif_text.encode("utf-8")).decode("utf-8")

        logger.info(
            "Crystal generation complete: %s, SG=%s (#%d), energy=%.4f eV",
            formula,
            final_sg_sym,
            final_sg_num,
            final_energy,
        )

        resp: Dict[str, Any] = {
            "formula": formula,
            "requested_space_group": space_group,
            "selected_space_group": best_sg,
            "space_group_randomly_selected": was_randomly_selected,
            "requested_space_group_symbol": requested_space_group_symbol,
            "final_space_group": final_sg_num,
            "final_space_group_symbol": final_sg_sym,
            "space_group_changed": final_sg_num != best_sg,
            "num_trials": num_trials,
            "best_crystal_energy": final_energy,
            "best_crystal_score": best_score,
            "score_breakdown": best_breakdown,
            "geometry_optimized": optimize_geometry,
            "symmetry_refined": refine_symmetry and optimize_geometry,
            "generation_stats": generation_stats,
            "cif_content": cif_text,
            "cif_base64": cif64,
            "filename": filename,
            "name": name,
            "description": description,
        }
        if optimize_geometry:
            resp["optimization_steps"] = optimization_steps
        return resp

    def _generate_crystal_iterative(
        self,
        formula: str,
        max_iterations: int,
        num_trials: int,
        convergence_threshold: float,
        top_k_spacegroups: int,
        symmetry_weight: float,
        symmetry_bias: float,
        crystal_systems: Optional[List[str]],
        min_distance_filter: float,
        volume_bounds: Tuple[float, float],
        preserve_symmetry: bool,
        optimization_max_steps: int,
        optimization_fmax: float,
        trajectory_interval: int,
    ) -> Dict[str, Any]:
        """Internal iterative crystal generation (called when max_iterations > 1)."""
        logger.info(
            "Starting iterative generation for %s (max_iterations=%d)",
            formula,
            max_iterations,
        )

        elements, counts = parse_chemical_formula(formula)
        compatible = self.get_compatible_space_groups(elements, counts)
        if not compatible:
            raise ValueError(
                f"No compatible space groups found for composition {formula}"
            )

        iteration_history: List[Dict[str, Any]] = []
        best_structure: Optional[Structure] = None
        best_energy = float("inf")
        best_sg = None
        detected_sg = None
        converged = False

        for iteration in range(max_iterations):
            logger.info(
                "=== Iteration %d/%d ===",
                iteration + 1,
                max_iterations,
            )

            # Determine space groups to try
            if iteration == 0:
                # First iteration: try top K space groups
                target_sg = None  # Will use multi_spacegroup mode
                logger.info(
                    "First iteration: exploring top %d space groups",
                    top_k_spacegroups,
                )
            else:
                # Later iterations: use detected space group from previous iteration
                target_sg = detected_sg
                logger.info(
                    "Iteration %d: regenerating in detected space group %d",
                    iteration + 1,
                    target_sg,
                )

            # Generate and optimize (single-shot, max_iterations=1)
            try:
                result = self.generate_crystal(
                    formula=formula,
                    space_group=target_sg,
                    num_trials=num_trials,
                    optimize_geometry=True,
                    multi_spacegroup=(iteration == 0),
                    top_k_spacegroups=top_k_spacegroups,
                    symmetry_weight=symmetry_weight,
                    symmetry_bias=symmetry_bias,
                    crystal_systems=crystal_systems if iteration == 0 else None,
                    refine_symmetry=not preserve_symmetry,
                    min_distance_filter=min_distance_filter,
                    volume_bounds=volume_bounds,
                    preserve_symmetry=preserve_symmetry,
                    max_iterations=1,  # Single-shot for each iteration
                    optimization_max_steps=optimization_max_steps,
                    optimization_fmax=optimization_fmax,
                    trajectory_interval=trajectory_interval,
                )
            except Exception as e:
                logger.warning("Iteration %d failed: %s", iteration + 1, e)
                if iteration == 0:
                    raise  # First iteration must succeed
                break

            current_energy = result["best_crystal_energy"]
            current_sg = result["final_space_group"]
            current_structure = self.get_structure()

            # Log iteration results
            energy_delta = (
                best_energy - current_energy if best_energy != float("inf") else 0
            )
            logger.info(
                "Iteration %d result: SG=%d (%s), energy=%.4f eV (Δ=%.4f eV)",
                iteration + 1,
                current_sg,
                result["final_space_group_symbol"],
                current_energy,
                energy_delta,
            )

            # Record iteration
            iteration_history.append(
                {
                    "iteration": iteration + 1,
                    "target_space_group": target_sg,
                    "final_space_group": current_sg,
                    "final_space_group_symbol": result["final_space_group_symbol"],
                    "energy": current_energy,
                    "energy_delta": energy_delta,
                    "optimization_steps": result.get("optimization_steps", 0),
                    "generation_stats": result.get("generation_stats", {}),
                }
            )

            # Check for improvement
            if current_energy < best_energy:
                improvement = best_energy - current_energy
                logger.info(
                    "New best structure found! Energy improved by %.4f eV",
                    improvement,
                )
                best_structure = current_structure
                best_energy = current_energy
                best_sg = current_sg

                # Check convergence
                if improvement < convergence_threshold and iteration > 0:
                    logger.info(
                        "Converged: energy improvement %.4f eV < threshold %.4f eV",
                        improvement,
                        convergence_threshold,
                    )
                    converged = True
                    break
            else:
                logger.info(
                    "No improvement in iteration %d (energy %.4f vs best %.4f)",
                    iteration + 1,
                    current_energy,
                    best_energy,
                )
                # No improvement - maybe we found a local minimum
                if iteration > 0:
                    logger.info("Stopping: no improvement after refinement")
                    break

            # Detect symmetry for next iteration
            try:
                spg = SpacegroupAnalyzer(current_structure, symprec=SYMPREC)
                detected_sg = spg.get_space_group_number()
                logger.info(
                    "Detected symmetry for next iteration: SG %d (%s)",
                    detected_sg,
                    spg.get_space_group_symbol(),
                )
            except Exception as e:
                logger.warning("Symmetry detection failed: %s", e)
                detected_sg = best_sg

        # Set final structure
        if best_structure is not None:
            self.set_structure(best_structure, add_to_trajectory=True)

        # Final analysis
        final_spg = SpacegroupAnalyzer(best_structure, symprec=SYMPREC)
        final_sg_num = final_spg.get_space_group_number()
        final_sg_sym = final_spg.get_space_group_symbol()

        # Generate CIF
        cif_text = str(CifWriter(best_structure, symprec=SYMPREC))
        cif64 = base64.b64encode(cif_text.encode("utf-8")).decode("utf-8")

        filename = f"{formula}_{final_sg_sym.replace('/', '-')}.cif"
        name = f"{formula} ({final_sg_sym})"
        description = (
            f"{formula} (iterative: {len(iteration_history)} iterations, "
            f"{'converged' if converged else 'max iterations'}, "
            f"final SG: {final_sg_sym} #{final_sg_num})"
        )

        logger.info(
            "Iterative generation complete: %d iterations, %s, final SG=%s (#%d), energy=%.4f eV",
            len(iteration_history),
            "converged" if converged else "max iterations reached",
            final_sg_sym,
            final_sg_num,
            best_energy,
        )

        return {
            "formula": formula,
            "final_space_group": final_sg_num,
            "final_space_group_symbol": final_sg_sym,
            "best_crystal_energy": best_energy,
            "converged": converged,
            "num_iterations": len(iteration_history),
            "iteration_history": iteration_history,
            "cif_content": cif_text,
            "cif_base64": cif64,
            "filename": filename,
            "name": name,
            "description": description,
        }

    # -------------------- Structure description / IO --------------------

    def _describe_structure(self, structure: Structure) -> Dict[str, Any]:
        """Get detailed description (composition, space group, lattice, Wyckoff sites)."""
        # Refine via CIF roundtrip to stabilize symmetry
        writer = CifWriter(
            structure,
            symprec=SYMPREC,
            write_magmoms=False,
            significant_figures=8,
            write_site_properties=False,
            refine_struct=True,
        )
        refined_structure = Structure.from_str(str(writer), fmt="cif")

        # Build PyXtal object for Wyckoff info
        atoms = _atoms_from_structure(refined_structure)
        crystal = pyxtal()
        crystal.from_seed(atoms)
        if not crystal.valid:
            raise ValueError(
                "Failed to create valid pyxtal structure from input structure"
            )

        space_group_number = crystal.group.number
        group = Group(space_group_number, dim=3)

        a, b, c, alpha, beta, gamma = crystal.lattice.get_para()
        composition = refined_structure.composition.reduced_formula

        wyckoff_sites = []
        for site in crystal.atom_sites:
            element = str(site.specie)
            coords = site.position if hasattr(site, "position") else [0.0, 0.0, 0.0]
            wp = site.wp
            label = f"{wp.multiplicity}{wp.letter}" if wp else "unknown"
            # Ensure site symmetry is computed (pyxtal populates wp.site_symm)
            try:
                wp.get_site_symmetry()
                site_symm = wp.site_symm
            except Exception:
                site_symm = "?"
            wyckoff_sites.append(
                {
                    "element": element,
                    "coordinates": [float(x) for x in coords],
                    "wyckoff_position": label,
                    "site_symmetry": site_symm,
                }
            )

        return {
            "composition": composition,
            "space_group": {
                "number": space_group_number,
                "symbol": group.symbol,
                "name": group.symbol,
            },
            "lattice": {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "crystal_system": getattr(group, "lattice_type", None),
            },
            "wyckoff_sites": wyckoff_sites,
        }

    def describe_crystal_from_file(
        self, file_url: str, filename: str
    ) -> Dict[str, Any]:
        """Get detailed description of a crystal structure from a file URL."""
        resp = requests.get(file_url, timeout=30)
        resp.raise_for_status()

        ext = Path(filename).suffix.lstrip(".").lower()
        atoms = ase_read(io.BytesIO(resp.content), format=ext or None)
        if isinstance(atoms, list):
            atoms = atoms[0]
        atoms.pbc = True

        structure = _atoms_to_structure_wrapped(atoms)
        return self._describe_structure(structure)

    def create_cif_from_description(
        self, crystal_description: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a CIF file from a crystal description object."""
        # Canonical structure build
        try:
            structure = _structure_from_description_like(crystal_description)
        except Exception as e:
            raise ValueError(f"Failed to reconstruct crystal structure: {e}")

        # CIF text (symmetry-bearing)
        cif_text = str(CifWriter(structure, symprec=SYMPREC))

        # Symmetry enrichment
        symmetry_info: Optional[Dict[str, Any]] = None
        actual_comp = None
        sg_symbol = None
        sg_number = None

        try:
            structure = Structure.from_str(cif_text, fmt="cif")
            spg = SpacegroupAnalyzer(structure, symprec=SYMPREC)
            symmetry_info = {
                "space_group_number": spg.get_space_group_number(),
                "space_group_symbol": spg.get_space_group_symbol(),
                "crystal_system": spg.get_crystal_system(),
                "point_group": spg.get_point_group_symbol(),
                "is_centrosymmetric": spg.is_laue(),
                "composition": structure.composition.reduced_formula,
            }
            actual_comp = symmetry_info["composition"]
            sg_symbol = symmetry_info["space_group_symbol"]
            sg_number = symmetry_info["space_group_number"]
        except Exception:
            actual_comp = crystal_description.get("composition", "unknown")
            sg_symbol = crystal_description["space_group"].get("symbol", "?")
            sg_number = crystal_description["space_group"].get("number", "?")

        cif64 = base64.b64encode(cif_text.encode("utf-8")).decode("utf-8")
        filename = f"{actual_comp}_{str(sg_symbol).replace('/', '-')}.cif"
        name = f"{actual_comp} ({sg_symbol})"
        description = (
            f"Crystal from description (space group: {sg_symbol} #{sg_number}, "
            f"crystal system: {symmetry_info['crystal_system'] if symmetry_info else crystal_description['lattice'].get('crystal_system', 'unknown')}, "
            f"point group: {symmetry_info['point_group'] if symmetry_info else 'unknown'})"
        )

        result: Dict[str, Any] = {
            "name": name,
            "description": description,
            "composition": actual_comp,
            "space_group": crystal_description["space_group"],
            "cif_content": cif_text,
            "cif_base64": cif64,
            "filename": filename,
        }
        if symmetry_info:
            result["symmetry_info"] = symmetry_info
        return result

    # -------------------- Crystal mutation methods --------------------

    def mutate_crystal(
        self,
        operations: List[Dict[str, Any]],
        repair: bool = True,
        min_distance: float = 0.8,
    ) -> None:
        """Apply a sequence of mutation operations to the internal crystal structure.

        Each operation is tracked individually as a separate trajectory frame.
        """
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )

        if self.enable_trajectory:
            self._add_trajectory_frame(
                structure=self._current_structure,
                frame_type="mutation_sequence_start",
                operation="mutate_crystal",
                parameters={
                    "operations": operations,
                    "repair": repair,
                    "min_distance": min_distance,
                },
                metadata={
                    "step": "sequence_start",
                    "total_operations": len(operations),
                },
            )

        current = self._current_structure.copy()

        # Apply each operation individually and track as separate frame
        for i, op_dict in enumerate(operations):
            if "op" not in op_dict:
                raise ValueError(f"Operation dict missing 'op': {op_dict}")

            op_name = op_dict["op"]
            params = {k: v for k, v in op_dict.items() if k != "op"}

            if not hasattr(self.crystal_ops, op_name):
                raise ValueError(f"Unknown operation: {op_name}")

            method = getattr(self.crystal_ops, op_name)

            try:
                # Track before applying operation
                if self.enable_trajectory:
                    self._add_trajectory_frame(
                        structure=current,
                        frame_type="mutation_operation_start",
                        operation=op_name,
                        parameters=params,
                        metadata={
                            "step": f"op_{i+1}_start",
                            "operation_index": i,
                            "operation_name": op_name,
                            "total_operations": len(operations),
                        },
                    )

                # Apply the operation
                current = method(current, **params)

                # Track after applying operation
                if self.enable_trajectory:
                    self._add_trajectory_frame(
                        structure=current,
                        frame_type="mutation_operation_applied",
                        operation=op_name,
                        parameters=params,
                        metadata={
                            "step": f"op_{i+1}_applied",
                            "operation_index": i,
                            "operation_name": op_name,
                            "total_operations": len(operations),
                        },
                    )

            except Exception as e:
                logger.error("Error during operation %s (index %d): %s", op_name, i, e)
                logger.debug("Operation parameters: %s", params)
                logger.debug("Current structure sites: %d", len(current))
                raise

        # Track end of sequence
        if self.enable_trajectory:
            self._add_trajectory_frame(
                structure=current,
                frame_type="mutation_sequence_applied",
                operation="mutate_crystal",
                parameters={
                    "operations": operations,
                    "repair": repair,
                    "min_distance": min_distance,
                },
                metadata={"step": "sequence_applied", "repair_applied": False},
            )

        # Apply repair if requested
        if repair:
            repaired = self.crystal_ops.repair_structure(current, min_distance)
            if self.enable_trajectory:
                self._add_trajectory_frame(
                    structure=repaired,
                    frame_type="mutation_sequence_repaired",
                    operation="mutate_crystal",
                    parameters={
                        "operations": operations,
                        "repair": repair,
                        "min_distance": min_distance,
                    },
                    metadata={"step": "sequence_repaired", "repair_applied": True},
                )
            current = repaired

        self.set_structure(current, add_to_trajectory=False)

    def scale_lattice(
        self,
        scale_factor: Union[float, Tuple[float, float, float]],
        isotropic: bool = True,
    ) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )

        if self.enable_trajectory:
            self._add_trajectory_frame(
                structure=self._current_structure,
                frame_type="mutation_start",
                operation="scale_lattice",
                parameters={"scale_factor": scale_factor, "isotropic": isotropic},
                metadata={"step": "pre_scale"},
            )

        mutated = self.crystal_ops.scale_lattice(
            self._current_structure, scale_factor, isotropic
        )

        if self.enable_trajectory:
            self._add_trajectory_frame(
                structure=mutated,
                frame_type="mutation_applied",
                operation="scale_lattice",
                parameters={"scale_factor": scale_factor, "isotropic": isotropic},
                metadata={"step": "post_scale"},
            )

        self.set_structure(mutated, add_to_trajectory=False)

    def shear_lattice(self, angle_deltas: Tuple[float, float, float]) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        if self.enable_trajectory:
            self._add_trajectory_frame(
                structure=self._current_structure,
                frame_type="mutation_start",
                operation="shear_lattice",
                parameters={"angle_deltas": angle_deltas},
                metadata={"step": "pre_shear"},
            )
        mutated = self.crystal_ops.shear_lattice(self._current_structure, angle_deltas)
        if self.enable_trajectory:
            self._add_trajectory_frame(
                structure=mutated,
                frame_type="mutation_applied",
                operation="shear_lattice",
                parameters={"angle_deltas": angle_deltas},
                metadata={"step": "post_shear"},
            )
        self.set_structure(mutated, add_to_trajectory=False)

    def change_space_group(
        self, target_space_group: int, symprec: float = SYMPREC
    ) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.change_space_group(
            self._current_structure, target_space_group, symprec
        )
        self.set_structure(mutated, add_to_trajectory=True)

    def symmetry_break(
        self,
        displacement_scale: float = 0.01,
        angle_perturbation: float = 0.1,
    ) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.symmetry_break(
            self._current_structure, displacement_scale, angle_perturbation
        )
        self.set_structure(mutated, add_to_trajectory=True)

    def substitute(
        self,
        element_from: str,
        element_to: str,
        fraction: float = 1.0,
    ) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.substitute(
            self._current_structure, element_from, element_to, fraction
        )
        self.set_structure(mutated, add_to_trajectory=True)

    def add_site(
        self,
        element: str,
        coordinates: Tuple[float, float, float],
        coords_are_cartesian: bool = False,
    ) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.add_site(
            self._current_structure, element, coordinates, coords_are_cartesian
        )
        self.set_structure(mutated, add_to_trajectory=True)

    def remove_site(
        self,
        site_indices: Optional[List[int]] = None,
        element: Optional[str] = None,
        max_remove: Optional[int] = None,
    ) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.remove_site(
            self._current_structure, site_indices, element, max_remove
        )
        self.set_structure(mutated, add_to_trajectory=True)

    def move_site(
        self,
        site_index: int,
        displacement: Optional[Tuple[float, float, float]] = None,
        new_coordinates: Optional[Tuple[float, float, float]] = None,
        coords_are_cartesian: bool = False,
    ) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.move_site(
            self._current_structure,
            site_index,
            displacement,
            new_coordinates,
            coords_are_cartesian,
        )
        self.set_structure(mutated, add_to_trajectory=True)

    def jitter_sites(self, sigma: float = 0.01, element: Optional[str] = None) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.jitter_sites(self._current_structure, sigma, element)
        self.set_structure(mutated, add_to_trajectory=True)

    def prune_overlaps(self, min_distance: float = 1.0) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.prune_overlaps(self._current_structure, min_distance)
        self.set_structure(mutated, add_to_trajectory=True)

    def supercell(
        self,
        scaling_matrix: Union[int, Tuple[int, int, int], List[List[int]]],
    ) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.supercell(self._current_structure, scaling_matrix)
        self.set_structure(mutated, add_to_trajectory=True)

    def shrink_to_primitive(self) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.shrink_to_primitive(self._current_structure)
        self.set_structure(mutated, add_to_trajectory=True)

    def repair_structure(self, min_distance: float = 0.8) -> None:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        mutated = self.crystal_ops.repair_structure(
            self._current_structure, min_distance
        )
        self.set_structure(mutated, add_to_trajectory=True)

    def validate_structure(self) -> bool:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        return self.crystal_ops.validate_structure(self._current_structure)

    # -------------------- Structure management --------------------

    def set_structure(
        self, structure: Structure, add_to_trajectory: bool = True
    ) -> None:
        """Set the internal structure and update PyXtal object."""
        self._current_structure = structure.copy()

        try:
            atoms = _atoms_from_structure(structure)
            self._current_pyxtal = pyxtal()
            self._current_pyxtal.from_seed(atoms)
        except Exception as e:
            logger.warning("Could not create pyxtal object from structure: %s", e)
            self._current_pyxtal = None

        if add_to_trajectory and self.enable_trajectory:
            frame_type = (
                "initial"
                if len(self._trajectory_structures) == 0
                else "structure_update"
            )
            self._add_trajectory_frame(
                structure=structure,
                frame_type=frame_type,
                metadata={"source": "set_structure"},
            )

    def get_structure(self) -> Optional[Structure]:
        return (
            self._current_structure.copy()
            if self._current_structure is not None
            else None
        )

    def load_structure_from_file(self, file_url: str, filename: str) -> None:
        resp = requests.get(file_url, timeout=30)
        resp.raise_for_status()

        ext = Path(filename).suffix.lstrip(".").lower()
        atoms = ase_read(io.BytesIO(resp.content), format=ext or None)
        if isinstance(atoms, list):
            atoms = atoms[0]
        atoms.pbc = True
        structure = _atoms_to_structure_wrapped(atoms)
        self.set_structure(structure)

    def from_json(self, crystal_data: Dict[str, Any]) -> None:
        """Load crystal structure from JSON representation and set as internal structure."""
        try:
            structure = _structure_from_description_like(crystal_data)
        except Exception as e:
            raise ValueError(f"Failed to reconstruct crystal structure from JSON: {e}")
        self.set_structure(structure)

    def to_json(self) -> Dict[str, Any]:
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )
        return self._describe_structure(self._current_structure)

    def has_structure(self) -> bool:
        return self._current_structure is not None

    def summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current structure.

        Returns:
            Dictionary containing structure composition, space group analysis,
            lattice parameters, and other structural properties.
        """
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )

        structure = self._current_structure

        # Basic composition and structure info
        composition = structure.composition
        formula = composition.reduced_formula
        num_sites = len(structure)
        volume = structure.volume
        density = structure.density

        # Lattice parameters
        lattice = structure.lattice
        a, b, c = lattice.lengths
        alpha, beta, gamma = lattice.angles

        # Space group analysis
        spg_info = None
        try:
            spg_analyzer = SpacegroupAnalyzer(structure, symprec=SYMPREC)
            spg_info = {
                "number": spg_analyzer.get_space_group_number(),
                "symbol": spg_analyzer.get_space_group_symbol(),
                "crystal_system": spg_analyzer.get_crystal_system(),
                "point_group": spg_analyzer.get_point_group_symbol(),
                "is_centrosymmetric": spg_analyzer.is_laue(),
                "symmetry_operations": len(spg_analyzer.get_symmetry_operations()),
            }
        except Exception as e:
            logger.warning("Space group analysis failed: %s", e)
            spg_info = {"error": str(e)}

        # Energy calculation if possible
        energy = None
        try:
            atoms = _atoms_from_structure(structure)
            atoms.calc = self.calculator
            energy = float(atoms.get_potential_energy())
        except Exception as e:
            logger.debug("Energy calculation failed: %s", e)

        # Element analysis
        element_counts = {}
        for site in structure:
            element = str(site.specie)
            element_counts[element] = element_counts.get(element, 0) + 1

        # Coordination analysis
        coord_info = None
        try:
            cnn = CrystalNN()
            coord_list = []
            for i in range(len(structure)):
                try:
                    cn = cnn.get_cn(structure, i)
                    coord_list.append(cn)
                except:
                    coord_list.append(0)

            coord_info = {
                "mean_coordination": float(np.mean(coord_list)),
                "std_coordination": float(np.std(coord_list)),
                "min_coordination": int(np.min(coord_list)),
                "max_coordination": int(np.max(coord_list)),
            }
        except Exception as e:
            logger.debug("Coordination analysis failed: %s", e)
            coord_info = {"error": str(e)}

        # Structure quality metrics
        try:
            # Check for overlapping atoms
            min_distances = []
            for i in range(len(structure)):
                for j in range(i + 1, len(structure)):
                    dist = structure.get_distance(i, j)
                    min_distances.append(dist)

            min_distance = min(min_distances) if min_distances else 0.0
            has_overlaps = min_distance < 0.5  # Arbitrary threshold

            quality_metrics = {
                "min_interatomic_distance": float(min_distance),
                "has_potential_overlaps": has_overlaps,
                "volume_per_atom": float(volume / num_sites),
            }
        except Exception as e:
            logger.debug("Quality metrics calculation failed: %s", e)
            quality_metrics = {"error": str(e)}

        # Trajectory info if available
        trajectory_info = None
        if self.enable_trajectory and self._trajectory_structures:
            trajectory_info = {
                "total_frames": len(self._trajectory_structures),
                "created_at": self._trajectory_info.get("created_at"),
                "random_seed": self._trajectory_info.get("random_seed"),
            }

        summary = {
            "formula": formula,
            "composition": {
                "reduced_formula": formula,
                "element_counts": element_counts,
                "total_sites": num_sites,
            },
            "lattice": {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
                "volume": float(volume),
                "density": float(density),
            },
            "space_group": spg_info,
            "coordination": coord_info,
            "quality_metrics": quality_metrics,
            "energy": energy,
            "trajectory": trajectory_info,
        }

        return summary

    # -------------------- Branching / comparison --------------------

    def copy(self) -> "GGen":
        """Create a deep copy of the GGen instance (without copying trajectory)."""
        new_ggen = GGen(
            calculator=self.calculator,
            random_seed=self.random_seed,
            enable_trajectory=self.enable_trajectory,
        )
        if self._current_structure is not None:
            new_ggen._current_structure = self._current_structure.copy()
        if self._current_pyxtal is not None:
            try:
                atoms = _atoms_from_structure(self._current_structure)
                new_ggen._current_pyxtal = pyxtal()
                new_ggen._current_pyxtal.from_seed(atoms)
            except Exception as e:
                logger.warning("Could not recreate pyxtal object in copy: %s", e)
                new_ggen._current_pyxtal = None
        new_ggen.crystal_ops = Operations(random_seed=self.random_seed)
        return new_ggen

    def branch(self, name: Optional[str] = None) -> "GGen":
        branched = self.copy()
        branched._branch_name = name or f"branch_{id(branched)}"
        return branched

    def get_branch_name(self) -> Optional[str]:
        return getattr(self, "_branch_name", None)

    def compare_branches(
        self, other_ggen: "GGen", method: str = "combined"
    ) -> Dict[str, Any]:
        if self._current_structure is None:
            raise ValueError("This GGen instance has no structure loaded")
        if other_ggen._current_structure is None:
            raise ValueError("The other GGen instance has no structure loaded")
        return self.calculate_similarity(other_ggen._current_structure, method=method)

    def calculate_similarity(
        self,
        other_structure: Union[Structure, str, dict],
        method: str = "fingerprint",
        tolerance: float = 0.1,
    ) -> Dict[str, Any]:
        """Calculate similarity between current structure and another structure.

        Args:
            other_structure: Structure to compare with (Structure, CIF string, or JSON dict)
            method: Similarity method - 'structural', 'compositional', 'symmetry', 'combined', or 'fingerprint'
            tolerance: Tolerance for structural comparisons (Å for lengths, ×10 for angles)

        Note: `tolerance` here is interpreted in Å for lattice lengths
        and scaled (×10) for angles, consistent with prior behavior.
        """
        if self._current_structure is None:
            raise ValueError(
                "No structure loaded. Use set_structure() or from_json() first."
            )

        # Parse other structure
        if isinstance(other_structure, str):
            try:
                other_struct = Structure.from_str(other_structure, fmt="cif")
            except Exception as e:
                raise ValueError(f"Failed to parse CIF string: {e}")
        elif isinstance(other_structure, dict):
            tmp = GGen()
            tmp.from_json(other_structure)
            other_struct = tmp.get_structure()
            if other_struct is None:
                raise ValueError("Failed to create structure from JSON")
        elif isinstance(other_structure, Structure):
            other_struct = other_structure
        else:
            raise ValueError(
                "other_structure must be a Structure, CIF string, or JSON dict"
            )

        current_struct = self._current_structure

        try:
            fp1 = get_structure_fingerprint(current_struct)
            fp2 = get_structure_fingerprint(other_struct)
            similarity = (1 - cosine(fp1, fp2)) * 100
            score = max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning("Fingerprint calculation failed: %s", e)
            score = 0.0

        return {
            "similarity": float(round(score, 2)),
            "score": float(round(score, 2) / 100),
        }
