"""
Crystal mutation operations for evolutionary crystal structure optimization.

This module implements the core mutation operations for crystal structures,
operating on pymatgen.core.Structure objects. All operations preserve fractional
coordinates and work with the crystal lattice and atomic sites.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import (
    PrimitiveCellTransformation,
    SupercellTransformation,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class MutationError(Exception):
    """Exception raised when crystal mutation operations fail."""

    pass


class Operations:
    """Collection of crystal mutation operations for evolutionary optimization."""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the crystal operations with optional random seed.

        Args:
            random_seed: Optional random seed for reproducible operations
        """
        # Avoid global RNG side effects; use instance RNGs.
        self.rng = np.random.default_rng(random_seed)
        self.py_rng = random.Random(random_seed)

    # ==================== LATTICE OPERATIONS ====================

    def scale_lattice(
        self,
        structure: Structure,
        scale_factor: Union[float, Tuple[float, float, float]],
        isotropic: bool = True,
    ) -> Structure:
        """Scale the lattice lengths.

        Args:
            structure: Input crystal structure
            scale_factor: Scaling factor(s). If isotropic=True, single float
                          meaning a,b,c lengths × factor (volume × factor³).
                          If isotropic=False, tuple of (a_scale, b_scale, c_scale).
            isotropic: Whether to scale all lattice vectors equally

        Returns:
            New structure with scaled lattice (fractional coordinates preserved)
        """
        new_structure = structure.copy()
        lat = new_structure.lattice

        if isotropic:
            if not isinstance(scale_factor, (float, int)):
                raise ValueError(
                    "For isotropic scaling, scale_factor must be a single float"
                )
            if scale_factor <= 0:
                raise ValueError("scale_factor must be > 0")
            # scale_lattice expects target volume
            new_volume = lat.volume * float(scale_factor) ** 3
            new_structure.scale_lattice(new_volume)
            return new_structure

        # Anisotropic scaling
        if not (isinstance(scale_factor, (tuple, list)) and len(scale_factor) == 3):
            raise ValueError(
                "For anisotropic scaling, scale_factor must be a tuple of 3 floats"
            )
        a_s, b_s, c_s = map(float, scale_factor)
        if a_s <= 0 or b_s <= 0 or c_s <= 0:
            raise ValueError("Anisotropic scale factors must be > 0")

        new_lattice = Lattice.from_parameters(
            lat.a * a_s, lat.b * b_s, lat.c * c_s, *lat.angles
        )
        new_structure = self._rebuild_with_lattice(new_structure, new_lattice)
        return new_structure

    def shear_lattice(
        self, structure: Structure, angle_deltas: Tuple[float, float, float]
    ) -> Structure:
        """Adjust cell angles by specified deltas.

        Args:
            structure: Input crystal structure
            angle_deltas: Tuple of (delta_alpha, delta_beta, delta_gamma) in degrees

        Returns:
            New structure with modified angles (fractional coordinates preserved)
        """
        if not (isinstance(angle_deltas, (tuple, list)) and len(angle_deltas) == 3):
            raise ValueError("angle_deltas must be a tuple/list of length 3")

        new_structure = structure.copy()
        lat = new_structure.lattice

        # Apply deltas and clamp to a physically reasonable range
        a0, b0, c0 = lat.a, lat.b, lat.c
        alpha, beta, gamma = lat.angles
        d_alpha, d_beta, d_gamma = map(float, angle_deltas)

        new_alpha = float(np.clip(alpha + d_alpha, 60.0, 120.0))
        new_beta = float(np.clip(beta + d_beta, 60.0, 120.0))
        new_gamma = float(np.clip(gamma + d_gamma, 60.0, 120.0))

        new_lattice = Lattice.from_parameters(
            a0, b0, c0, new_alpha, new_beta, new_gamma
        )
        new_structure = self._rebuild_with_lattice(new_structure, new_lattice)
        return new_structure

    # ==================== SYMMETRY OPERATIONS ====================

    def change_space_group(
        self, structure: Structure, target_space_group: int, symprec: float = 0.1
    ) -> Structure:
        """Re-express structure in target space group.

        Notes:
            Converting an arbitrary structure to a *different* space group is nontrivial.
            Here we standardize and refine; if the refined symmetry does not match the
            target, we raise MutationError (explicit contract).

        Args:
            structure: Input crystal structure
            target_space_group: Target space group number (1-230)
            symprec: Symmetry precision for analysis

        Returns:
            New standardized structure in the same or target space group

        Raises:
            MutationError if the target space group cannot be achieved by standardization.
        """
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        current_sg = analyzer.get_space_group_number()
        if current_sg == target_space_group:
            return structure.copy()

        # Try a sequence of standardizations and refinements
        candidates: List[Structure] = []
        try:
            candidates.append(analyzer.get_refined_structure())
        except Exception:
            pass
        try:
            candidates.append(analyzer.get_primitive_standard_structure())
        except Exception:
            pass
        try:
            candidates.append(analyzer.get_conventional_standard_structure())
        except Exception:
            pass

        for cand in candidates:
            try:
                sg = SpacegroupAnalyzer(cand, symprec=symprec).get_space_group_number()
                if sg == target_space_group:
                    return cand
            except Exception:
                continue

        raise MutationError(
            f"Cannot convert from space group {current_sg} to {target_space_group} "
            f"using standardization/refinement with symprec={symprec}"
        )

    def symmetry_break(
        self,
        structure: Structure,
        displacement_scale: float = 0.01,
        angle_perturbation: float = 0.1,
    ) -> Structure:
        """Intentionally lower symmetry with random displacements and small angle tweaks.

        Args:
            structure: Input crystal structure
            displacement_scale: Std. dev. for per-site fractional displacements
            angle_perturbation: Std. dev. for angle deltas in degrees

        Returns:
            New structure with lowered symmetry
        """
        new_structure = structure.copy()

        # Random fractional displacements
        for i in range(len(new_structure)):
            site = new_structure[i]
            disp = self.rng.normal(0.0, displacement_scale, size=3)
            new_coords = (site.frac_coords + disp) % 1.0
            new_structure.replace(i, site.specie, new_coords)

        # Small random angle perturbations
        lat = new_structure.lattice
        d_alpha, d_beta, d_gamma = self.rng.normal(0.0, angle_perturbation, size=3)
        alpha, beta, gamma = lat.angles
        new_alpha = float(np.clip(alpha + d_alpha, 60.0, 120.0))
        new_beta = float(np.clip(beta + d_beta, 60.0, 120.0))
        new_gamma = float(np.clip(gamma + d_gamma, 60.0, 120.0))

        new_lattice = Lattice.from_parameters(
            lat.a, lat.b, lat.c, new_alpha, new_beta, new_gamma
        )
        new_structure = self._rebuild_with_lattice(new_structure, new_lattice)

        return new_structure

    # ==================== COMPOSITION / SITE OPERATIONS ====================

    def substitute(
        self,
        structure: Structure,
        element_from: str,
        element_to: str,
        fraction: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> Structure:
        """Change species identity on selected sites.

        Args:
            structure: Input crystal structure
            element_from: Element symbol to replace
            element_to: Element symbol to replace with
            fraction: Fraction of sites to substitute (0.0-1.0)
            random_seed: Optional per-call seed (overrides instance RNG)

        Returns:
            New structure with substituted elements
        """
        if fraction < 0:
            raise ValueError("fraction must be >= 0")
        if fraction == 0:
            return structure.copy()

        rng = (
            np.random.default_rng(random_seed) if random_seed is not None else self.rng
        )
        new_structure = structure.copy()

        # Indices of sites matching element_from
        target_sites = [
            i for i, s in enumerate(new_structure) if s.specie.symbol == element_from
        ]
        if not target_sites:
            available = {s.specie.symbol for s in new_structure}
            raise MutationError(
                f"No sites found with element {element_from}. "
                f"Available elements: {sorted(available)}. "
                f"Structure has {len(new_structure)} sites."
            )

        n_targets = len(target_sites)
        n_sub = int(math.ceil(min(max(fraction, 0.0), 1.0) * n_targets))
        n_sub = min(n_sub, n_targets)
        if n_sub == 0:
            return new_structure

        sites_to_sub = rng.choice(target_sites, size=n_sub, replace=False)

        for idx in sites_to_sub:
            site = new_structure[idx]
            new_structure.replace(idx, element_to, site.frac_coords)

        return new_structure

    def add_site(
        self,
        structure: Structure,
        element: str,
        coordinates: Tuple[float, float, float],
        coords_are_cartesian: bool = False,
    ) -> Structure:
        """Insert a new atomic site.

        Args:
            structure: Input crystal structure
            element: Element symbol for new site
            coordinates: Position coordinates (x, y, z)
            coords_are_cartesian: Whether coordinates are Cartesian (True) or fractional (False)

        Returns:
            New structure with added site
        """
        new_structure = structure.copy()
        if coords_are_cartesian:
            frac = new_structure.lattice.get_fractional_coords(coordinates)
        else:
            frac = np.array(coordinates, dtype=float)
        frac = frac % 1.0
        new_structure.append(element, frac)
        return new_structure

    def remove_site(
        self,
        structure: Structure,
        site_indices: Optional[List[int]] = None,
        element: Optional[str] = None,
        max_remove: Optional[int] = None,
    ) -> Structure:
        """Delete one or more sites.

        Args:
            structure: Input crystal structure
            site_indices: Specific site indices to remove (if provided)
            element: Element to remove (if provided, removes all sites of this element)
            max_remove: Maximum number of sites to remove

        Returns:
            New structure with removed sites
        """
        new_structure = structure.copy()

        if site_indices is not None:
            to_remove = [idx for idx in site_indices if 0 <= idx < len(new_structure)]
            for idx in sorted(to_remove, reverse=True):
                new_structure.remove_sites([idx])
        elif element is not None:
            to_remove = [
                i for i, s in enumerate(new_structure) if s.specie.symbol == element
            ]
            if max_remove is not None:
                to_remove = to_remove[:max_remove]
            for idx in sorted(to_remove, reverse=True):
                new_structure.remove_sites([idx])
        else:
            raise ValueError("Must specify either site_indices or element to remove")

        if len(new_structure) == 0:
            raise MutationError("Cannot remove all sites from structure")

        return new_structure

    def move_site(
        self,
        structure: Structure,
        site_index: int,
        displacement: Optional[Tuple[float, float, float]] = None,
        new_coordinates: Optional[Tuple[float, float, float]] = None,
        coords_are_cartesian: bool = False,
    ) -> Structure:
        """Relocate a selected site.

        Args:
            structure: Input crystal structure
            site_index: Index of site to move
            displacement: Fractional displacement (dx, dy, dz)
            new_coordinates: New coordinates (fractional or Cartesian)
            coords_are_cartesian: Whether new_coordinates are Cartesian

        Returns:
            New structure with moved site
        """
        if not (0 <= site_index < len(structure)):
            raise MutationError(f"Invalid site index: {site_index}")

        new_structure = structure.copy()
        site = new_structure[site_index]

        if displacement is not None:
            new_coords = site.frac_coords + np.array(displacement, dtype=float)
        elif new_coordinates is not None:
            if coords_are_cartesian:
                new_coords = new_structure.lattice.get_fractional_coords(
                    new_coordinates
                )
            else:
                new_coords = np.array(new_coordinates, dtype=float)
        else:
            raise ValueError("Must specify either displacement or new_coordinates")

        new_coords = new_coords % 1.0
        new_structure.replace(site_index, site.specie, new_coords)
        return new_structure

    def jitter_sites(
        self,
        structure: Structure,
        sigma: float = 0.01,
        element: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> Structure:
        """Apply small random fractional displacements to sites.

        Args:
            structure: Input crystal structure
            sigma: Standard deviation of Gaussian displacements (fractional units)
            element: Specific element to jitter (if None, jitter all sites)
            random_seed: Optional per-call seed (overrides instance RNG)

        Returns:
            New structure with jittered sites
        """
        rng = (
            np.random.default_rng(random_seed) if random_seed is not None else self.rng
        )
        new_structure = structure.copy()

        if element is not None:
            indices = [
                i for i, s in enumerate(new_structure) if s.specie.symbol == element
            ]
        else:
            indices = list(range(len(new_structure)))

        for idx in indices:
            site = new_structure[idx]
            disp = rng.normal(0.0, sigma, size=3)
            new_coords = (site.frac_coords + disp) % 1.0
            new_structure.replace(idx, site.specie, new_coords)

        return new_structure

    def prune_overlaps(
        self, structure: Structure, min_distance: float = 1.0
    ) -> Structure:
        """Remove sites that violate minimum distance rules.

        Args:
            structure: Input crystal structure
            min_distance: Minimum allowed interatomic distance in Angstroms

        Returns:
            New structure with overlapping sites removed
        """
        if min_distance <= 0:
            raise ValueError("min_distance must be > 0")

        new_structure = structure.copy()
        to_remove: List[int] = []

        # O(N^2); acceptable for modest cell sizes. Uses PBC-aware metric.
        for i in range(len(new_structure)):
            for j in range(i + 1, len(new_structure)):
                d = new_structure.get_distance(i, j)
                if d < min_distance:
                    to_remove.append(max(i, j))

        for idx in sorted(set(to_remove), reverse=True):
            new_structure.remove_sites([idx])

        if len(new_structure) == 0:
            raise MutationError("All sites removed due to overlaps")

        return new_structure

    # ==================== CELL TOPOLOGY OPERATIONS ====================

    def supercell(
        self,
        structure: Structure,
        scaling_matrix: Union[int, Tuple[int, int, int], List[List[int]]],
    ) -> Structure:
        """Replicate the unit cell by integer multipliers."""
        transformation = SupercellTransformation(scaling_matrix=scaling_matrix)
        return transformation.apply_transformation(structure)

    def shrink_to_primitive(self, structure: Structure) -> Structure:
        """Convert to primitive cell."""
        transformation = PrimitiveCellTransformation()
        return transformation.apply_transformation(structure)

    # ==================== UTILITY METHODS ====================

    def get_random_site_index(
        self, structure: Structure, element: Optional[str] = None
    ) -> int:
        """Get a random site index from the structure (instance RNG)."""
        if element is not None:
            candidates = [
                i for i, s in enumerate(structure) if s.specie.symbol == element
            ]
            if not candidates:
                available = {s.specie.symbol for s in structure}
                raise MutationError(
                    f"No sites found with element {element}. "
                    f"Available elements: {sorted(available)}. "
                    f"Structure has {len(structure)} sites."
                )
            return self.py_rng.choice(candidates)
        return self.py_rng.randrange(len(structure))

    def get_random_coordinates(
        self, structure: Structure
    ) -> Tuple[float, float, float]:
        """Get random fractional coordinates within the unit cell (instance RNG)."""
        return tuple(self.rng.random(3))

    def validate_structure(self, structure: Structure) -> bool:
        """Validate that the structure is reasonable."""
        try:
            if len(structure) == 0:
                return False
            lat = structure.lattice
            if any(x <= 0 for x in (lat.a, lat.b, lat.c)):
                return False
            a, b, c = lat.angles
            if not (0 < a < 180 and 0 < b < 180 and 0 < c < 180):
                return False
            if lat.volume <= 0:
                return False
            return True
        except Exception:
            return False

    def repair_structure(
        self, structure: Structure, min_distance: float = 0.8
    ) -> Structure:
        """Repair common issues in the structure (overlaps + wrap coords)."""
        try:
            repaired = self.prune_overlaps(structure, min_distance)
        except MutationError:
            repaired = structure.copy()

        # Wrap to [0, 1)
        for i in range(len(repaired)):
            site = repaired[i]
            repaired.replace(i, site.specie, site.frac_coords % 1.0)

        return repaired

    def _rebuild_with_lattice(
        self, structure: Structure, lattice: Lattice
    ) -> Structure:
        """Recreate Structure with a new lattice, preserving species and fractional coords."""
        species = [s.specie for s in structure.sites]
        fracs = [s.frac_coords for s in structure.sites]
        try:
            site_properties = (
                structure.site_properties if structure.site_properties else None
            )
            return Structure(lattice, species, fracs, site_properties=site_properties)
        except Exception:
            return Structure(lattice, species, fracs)


# ==================== CONVENIENCE FUNCTIONS ====================


def create_mutation_ops(random_seed: Optional[int] = None) -> Operations:
    """Create an Operations instance with optional random seed."""
    return Operations(random_seed=random_seed)


def apply_mutation_sequence(
    structure: Structure,
    operations: List[Dict[str, Any]],
    random_seed: Optional[int] = None,
) -> Structure:
    """Apply a sequence of mutation operations to a structure."""
    ops = Operations(random_seed=random_seed)
    current = structure.copy()

    for op_dict in operations:
        if "op" not in op_dict:
            raise MutationError(f"Operation dict missing 'op': {op_dict}")
        op_name = op_dict["op"]
        params = {k: v for k, v in op_dict.items() if k != "op"}

        if not hasattr(ops, op_name):
            raise MutationError(f"Unknown operation: {op_name}")

        method = getattr(ops, op_name)
        try:
            current = method(current, **params)
        except Exception as e:
            raise MutationError(f"Operation {op_name} failed: {e}")
    return current
