from __future__ import annotations

import base64
import io
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
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
        self, compatible_groups: List[Dict[str, Any]], random_seed: Optional[int] = None
    ) -> int:
        """Select a random space group from compatible groups with preference for higher symmetry.

        Weighted by crystal system and modest bonuses (Wyckoff count, inversion, etc.).
        """
        if not compatible_groups:
            raise ValueError("No compatible space groups provided")

        rng = np.random.default_rng(
            self.random_seed if random_seed is None else random_seed
        )

        crystal_system_weights = {
            (1, 2): 1.0,  # triclinic
            (3, 15): 2.0,  # monoclinic
            (16, 74): 3.0,  # orthorhombic
            (75, 142): 4.0,  # tetragonal
            (143, 167): 4.5,  # trigonal
            (168, 194): 5.0,  # hexagonal
            (195, 230): 6.0,  # cubic
        }

        def _sys_weight(sg: int) -> float:
            for (a, b), w in crystal_system_weights.items():
                if a <= sg <= b:
                    return w
            return 1.0

        weights = []
        for g in compatible_groups:
            sg = g["number"]
            base = _sys_weight(sg)
            wyckoff_bonus = min(g.get("wyckoff_positions", 1) / 10.0, 1.0)
            chiral_bonus = 0.1 if g.get("chiral", False) else 0.0
            inversion_bonus = 0.2 if g.get("inversion", False) else 0.0
            sgnum_bonus = (sg / 230.0) * 0.5
            weights.append(
                base + wyckoff_bonus + chiral_bonus + inversion_bonus + sgnum_bonus
            )

        probs = np.array(weights, dtype=float)
        probs /= probs.sum()

        idx = rng.choice(len(compatible_groups), p=probs)
        return int(compatible_groups[idx]["number"])

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
    ) -> Tuple[Structure, float, int]:
        """Optimize geometry using ASE LBFGS with optional variable cell.

        Returns:
            (optimized_structure, final_energy_eV, steps)
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
                },
                metadata={"step": 0},
            )

        atoms = _atoms_from_structure(self._current_structure)
        atoms.calc = self.calculator

        dyn_target = FrechetCellFilter(atoms) if relax_cell else atoms

        if self.enable_trajectory and trajectory_interval > 0:

            class TrajectoryLBFGS(LBFGS):
                def __init__(self, atoms, ggen: GGen, interval: int, **kwargs):
                    super().__init__(atoms, **kwargs)
                    self.ggen = ggen
                    self.interval = interval
                    self.step_count = 0
                    self.adaptor = AseAtomsAdaptor()

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
                        structure = self.adaptor.get_structure(a)
                        self.ggen._add_trajectory_frame(
                            structure=structure,
                            frame_type="optimization_step",
                            operation="optimize_geometry",
                            metadata={"step": self.step_count},
                        )
                    return result

            optimizer = TrajectoryLBFGS(
                dyn_target, self, trajectory_interval, maxstep=0.2
            )
        else:
            optimizer = LBFGS(dyn_target, maxstep=0.2)

        try:
            optimizer.run(fmax=fmax, steps=max_steps)
            final_energy = atoms.get_potential_energy()
            num_steps = optimizer.get_number_of_steps()

            if self.enable_trajectory:
                final_struct = AseAtomsAdaptor().get_structure(atoms)
                self._add_trajectory_frame(
                    structure=final_struct,
                    frame_type="optimization_end",
                    operation="optimize_geometry",
                    parameters={
                        "max_steps": max_steps,
                        "fmax": fmax,
                        "relax_cell": relax_cell,
                        "steps_completed": num_steps,
                    },
                    metadata={"step": num_steps, "converged": True},
                )

            optimized_structure = AseAtomsAdaptor().get_structure(atoms)
            self.set_structure(optimized_structure, add_to_trajectory=False)
            return optimized_structure, float(final_energy), int(num_steps)

        except Exception as e:
            logger.error("Geometry optimization failed: %s", e)
            try:
                original_energy = float(atoms.get_potential_energy())
            except Exception:
                original_energy = float("nan")

            if self.enable_trajectory:
                fallback_struct = AseAtomsAdaptor().get_structure(atoms)
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
    ) -> Dict[str, Any]:
        """Generate a crystal structure using PyXtal, evaluate energies, and return the best.

        Returns metadata + CIF content (text and base64).
        """
        if num_trials < 1:
            raise ValueError("num_trials must be >= 1")
        num_trials = int(min(num_trials, 100))

        elements, counts = parse_chemical_formula(formula)

        selected_space_group = space_group
        was_randomly_selected = False

        if selected_space_group is None:
            compatible = self.get_compatible_space_groups(elements, counts)
            if not compatible:
                raise ValueError(
                    f"No compatible space groups found for composition {formula}"
                )
            selected_space_group = (
                self.select_random_space_group_with_symmetry_preference(
                    compatible, random_seed=self.random_seed
                )
            )
            was_randomly_selected = True
        else:
            try:
                g = Group(int(selected_space_group), dim=3)
                ok, _msg = g.check_compatible(counts)
                if not ok:
                    compatible = self.get_compatible_space_groups(elements, counts)
                    raise ValueError(
                        f"Composition {counts} not compatible with space group {selected_space_group}. "
                        f"Compatible space groups: {[x['number'] for x in compatible]}"
                    )
            except Exception as e:
                compatible = self.get_compatible_space_groups(elements, counts)
                raise ValueError(
                    f"Space group validation failed: {e}. "
                    f"Compatible space groups: {[x['number'] for x in compatible]}"
                )

        crystals: List[pyxtal] = []
        for _ in range(num_trials):
            c = pyxtal()
            # Pass seed when available for reproducibility
            try:
                c.from_random(
                    dim=3,
                    group=selected_space_group,
                    species=elements,
                    numIons=counts,
                    seed=self.random_seed,
                )
            except TypeError:
                # Older pyxtal versions may not accept seed
                c.from_random(
                    dim=3, group=selected_space_group, species=elements, numIons=counts
                )
            if c.valid:
                crystals.append(c)

        if not crystals:
            compatible = self.get_compatible_space_groups(elements, counts)
            raise ValueError(
                "Failed to generate valid crystal structure. "
                f"Compatible space groups: {[x['number'] for x in compatible]}"
            )

        # Energy evaluation and selection
        energies: List[float] = []
        for c in crystals:
            atoms = c.to_ase()
            atoms.calc = self.calculator
            energies.append(float(atoms.get_potential_energy()))

        best_idx = int(np.argmin(energies))
        best_crystal = crystals[best_idx]
        structure = best_crystal.to_pymatgen()
        final_energy = float(energies[best_idx])
        optimization_steps = 0

        if optimize_geometry:
            self.set_structure(structure, add_to_trajectory=True)
            structure, final_energy, optimization_steps = self.optimize_geometry(
                max_steps=400, fmax=0.01, relax_cell=True
            )

        # Set final structure and analyze symmetry
        self.set_structure(structure, add_to_trajectory=not optimize_geometry)
        spg = SpacegroupAnalyzer(structure, symprec=SYMPREC)
        final_sg_num = spg.get_space_group_number()
        final_sg_sym = spg.get_space_group_symbol()

        requested_space_group_symbol = best_crystal.group.symbol
        filename = f"{formula}_{final_sg_sym.replace('/', '-')}.cif"
        name = f"{formula} ({final_sg_sym})"

        optimization_text = ""
        if optimize_geometry:
            optimization_text = (
                f", optimized: {optimization_steps} steps, cell relaxed (isotropic)"
            )

        if optimize_geometry and final_sg_num != selected_space_group:
            if was_randomly_selected:
                description = (
                    f"{formula} (auto-selected SG: {requested_space_group_symbol} #{selected_space_group}, "
                    f"calculated SG: {final_sg_sym} #{final_sg_num}{optimization_text})"
                )
            else:
                description = (
                    f"{formula} (requested SG: {requested_space_group_symbol} #{selected_space_group}, "
                    f"calculated SG: {final_sg_sym} #{final_sg_num}{optimization_text})"
                )
        else:
            sg_source = "auto-selected" if was_randomly_selected else "requested"
            description = f"{formula} ({sg_source} space group: {final_sg_sym} #{final_sg_num}{optimization_text})"

        # Produce CIF content using CifWriter (ensures clean symmetry-bearing CIF)
        cif_text = str(CifWriter(structure, symprec=SYMPREC))
        cif64 = base64.b64encode(cif_text.encode("utf-8")).decode("utf-8")

        resp: Dict[str, Any] = {
            "formula": formula,
            "requested_space_group": space_group,
            "selected_space_group": selected_space_group,
            "space_group_randomly_selected": was_randomly_selected,
            "requested_space_group_symbol": requested_space_group_symbol,
            "final_space_group": final_sg_num,
            "final_space_group_symbol": final_sg_sym,
            "space_group_changed": final_sg_num != selected_space_group,
            "num_trials": num_trials,
            "best_crystal_energy": final_energy,
            "geometry_optimized": optimize_geometry,
            "cif_content": cif_text,
            "cif_base64": cif64,
            "filename": filename,
            "name": name,
            "description": description,
        }
        if optimize_geometry:
            resp["optimization_steps"] = optimization_steps
        return resp

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

        structure = AseAtomsAdaptor().get_structure(atoms)
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
        structure = AseAtomsAdaptor().get_structure(atoms)
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
