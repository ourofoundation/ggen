"""
Phonon calculations and dynamical stability testing for GGen.

This module provides functionality for:
- Computing phonon band structures using finite-displacement method
- Detecting imaginary modes (dynamical instability)
- Batch testing multiple candidate structures for stability

Uses Phonopy for phonon calculations with ASE calculators.
"""

from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.core import Structure

# Suppress spglib deprecation warnings (phonopy uses old dict interface internally)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
# Suppress seekpath deprecation warnings (phonopy uses old dict interface internally)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="seekpath")

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class PhononResult:
    """Result of a phonon calculation.

    Attributes:
        is_stable: True if no imaginary modes detected (dynamically stable).
        num_imaginary_modes: Number of phonon modes with imaginary frequencies.
        min_frequency: Minimum phonon frequency (THz). Negative = imaginary.
        max_frequency: Maximum phonon frequency (THz).
        min_imaginary_frequency: Most negative frequency (THz), or None if stable.
        frequencies: Full array of phonon frequencies if requested.
        band_structure_plot: Base64-encoded PNG of band structure, if generated.
        supercell: Supercell dimensions used for calculation.
        metadata: Additional calculation metadata.
    """

    is_stable: bool
    num_imaginary_modes: int = 0
    min_frequency: Optional[float] = None
    max_frequency: Optional[float] = None
    min_imaginary_frequency: Optional[float] = None
    frequencies: Optional[np.ndarray] = None
    band_structure_plot: Optional[str] = None
    supercell: Tuple[int, int, int] = (3, 3, 3)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StabilityTestResult:
    """Result of testing multiple candidates for dynamical stability.

    Attributes:
        stable_candidate_idx: Index of first stable candidate, or None if all unstable.
        stable_structure: The first dynamically stable structure found.
        phonon_result: PhononResult for the stable structure.
        all_results: List of (index, PhononResult) for all tested candidates.
        num_tested: Number of candidates tested before finding stable one.
    """

    stable_candidate_idx: Optional[int] = None
    stable_structure: Optional[Structure] = None
    phonon_result: Optional[PhononResult] = None
    all_results: List[Tuple[int, PhononResult]] = field(default_factory=list)
    num_tested: int = 0


# -----------------------------------------------------------------------------
# Converters between ASE/pymatgen and Phonopy
# -----------------------------------------------------------------------------


def _structure_to_phonopy_atoms(structure: Structure) -> PhonopyAtoms:
    """Convert pymatgen Structure to PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=[str(s) for s in structure.species],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
    )


def _ase_to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE Atoms to PhonopyAtoms."""
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    )


def _phonopy_atoms_to_ase(patoms: PhonopyAtoms) -> Atoms:
    """Convert PhonopyAtoms to ASE Atoms."""
    return Atoms(
        symbols=patoms.symbols,
        cell=patoms.cell,
        scaled_positions=patoms.scaled_positions,
        pbc=True,
    )


def _structure_to_ase(structure: Structure) -> Atoms:
    """Convert pymatgen Structure to ASE Atoms."""
    return Atoms(
        symbols=[str(s) for s in structure.species],
        cell=structure.lattice.matrix,
        scaled_positions=structure.frac_coords,
        pbc=True,
    )


# -----------------------------------------------------------------------------
# Phonon Calculation Functions
# -----------------------------------------------------------------------------


def calculate_phonons(
    structure: Union[Structure, Atoms],
    calculator,
    supercell: Tuple[int, int, int] = (3, 3, 3),
    displacement_distance: float = 0.01,
    imaginary_threshold: float = -0.3,
    generate_plot: bool = False,
    log_callback: Optional[Callable[[str, str], None]] = None,
) -> PhononResult:
    """Calculate phonon properties and check for dynamical stability.

    This uses the finite-displacement method with Phonopy to compute
    phonon frequencies across the Brillouin zone and detect imaginary
    modes that indicate dynamical instability.

    Args:
        structure: Crystal structure (pymatgen Structure or ASE Atoms).
        calculator: ASE calculator for force calculations.
        supercell: Supercell dimensions for phonon calculation.
            Larger supercells give more accurate phonons but are more expensive.
            Default (3,3,3) is usually sufficient for stability checks.
        displacement_distance: Atomic displacement distance in Å.
        imaginary_threshold: Frequency threshold (THz) below which modes
            are considered imaginary. Default -0.3 THz accounts for
            numerical noise near Gamma point.
        generate_plot: If True, generate band structure plot (slower).
        log_callback: Optional callback for progress logging.
            Signature: log_callback(message: str, level: str)

    Returns:
        PhononResult with stability information.

    Raises:
        ImportError: If phonopy is not installed.
        RuntimeError: If phonon calculation fails.
    """

    def log(msg: str, level: str = "info"):
        if log_callback:
            log_callback(msg, level)
        if level == "info":
            logger.info(msg)
        elif level == "warn":
            logger.warning(msg)
        elif level == "debug":
            logger.debug(msg)

    # Convert structure to appropriate format
    if isinstance(structure, Structure):
        phonopy_atoms = _structure_to_phonopy_atoms(structure)
        ase_template = _structure_to_ase(structure)
    else:
        phonopy_atoms = _ase_to_phonopy_atoms(structure)
        ase_template = structure.copy()

    formula = ase_template.get_chemical_formula()
    log(f"Starting phonon calculation for {formula}")

    # Initialize Phonopy
    log(f"Using supercell: {supercell}")
    phonon = Phonopy(
        phonopy_atoms,
        supercell_matrix=np.diag(supercell),
        primitive_matrix="auto",
    )

    # Generate displacements
    log("Generating atomic displacements...")
    phonon.generate_displacements(distance=displacement_distance)

    # Calculate forces for each displaced supercell
    supercells = phonon.supercells_with_displacements
    num_displacements = len(supercells)
    log(f"Calculating forces for {num_displacements} displacements...")

    forces = []
    for idx, scell in enumerate(supercells):
        ase_scell = _phonopy_atoms_to_ase(scell)
        ase_scell.calc = calculator
        forces.append(ase_scell.get_forces())

        if (idx + 1) % 10 == 0 or idx == num_displacements - 1:
            log(f"  Displacement {idx + 1}/{num_displacements}", level="debug")

    # Set forces and compute force constants
    phonon.forces = forces
    log("Computing force constants...")
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    # Run band structure calculation
    log("Computing phonon band structure...")
    phonon.auto_band_structure(plot=False)

    # Extract frequencies and analyze
    band_dict = phonon.get_band_structure_dict()
    all_freqs = np.concatenate(
        [np.asarray(f).flatten() for f in band_dict["frequencies"]], axis=None
    )

    # Detect imaginary modes
    imaginary_mask = all_freqs < imaginary_threshold
    num_imaginary = int(np.sum(imaginary_mask))
    is_stable = num_imaginary == 0

    min_freq = float(all_freqs.min())
    max_freq = float(all_freqs.max())
    min_imag_freq = (
        float(all_freqs[imaginary_mask].min()) if num_imaginary > 0 else None
    )

    # Generate plot if requested
    band_plot_b64 = None
    if generate_plot:
        try:
            import base64

            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = phonon.auto_band_structure(plot=True).gcf()
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            band_plot_b64 = base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            log(f"Failed to generate band structure plot: {e}", level="warn")

    stability_msg = "dynamically stable" if is_stable else "dynamically UNSTABLE"
    log(f"Result: {formula} is {stability_msg}")
    if not is_stable:
        log(
            f"  {num_imaginary} imaginary modes, min frequency: {min_imag_freq:.3f} THz",
            level="warn",
        )

    return PhononResult(
        is_stable=is_stable,
        num_imaginary_modes=num_imaginary,
        min_frequency=round(min_freq, 4),
        max_frequency=round(max_freq, 4),
        min_imaginary_frequency=round(min_imag_freq, 4) if min_imag_freq else None,
        frequencies=all_freqs,
        band_structure_plot=band_plot_b64,
        supercell=supercell,
        metadata={
            "num_displacements": num_displacements,
            "displacement_distance": displacement_distance,
            "imaginary_threshold": imaginary_threshold,
        },
    )


def check_dynamical_stability(
    structure: Union[Structure, Atoms],
    calculator,
    supercell: Tuple[int, int, int] = (2, 2, 2),
    quick: bool = True,
) -> bool:
    """Quick check for dynamical stability.

    A convenience wrapper around calculate_phonons that just returns
    True/False for stability without additional details.

    Args:
        structure: Crystal structure to check.
        calculator: ASE calculator for force calculations.
        supercell: Supercell dimensions.
        quick: If True, use smaller supercell for faster (but less accurate) check.

    Returns:
        True if structure is dynamically stable (no imaginary modes).
    """
    if quick:
        # Use smaller supercell for quick stability check
        supercell = (2, 2, 2)

    result = calculate_phonons(
        structure=structure,
        calculator=calculator,
        supercell=supercell,
        generate_plot=False,
    )
    return result.is_stable


def find_first_stable_candidate(
    candidates: List[Tuple[Structure, float]],
    calculator,
    supercell: Tuple[int, int, int] = (2, 2, 2),
    max_candidates: Optional[int] = None,
    log_callback: Optional[Callable[[str, str], None]] = None,
) -> StabilityTestResult:
    """Find the first dynamically stable candidate from a ranked list.

    Tests candidates in order (assumed to be sorted by energy, best first)
    until a dynamically stable one is found.

    Args:
        candidates: List of (structure, energy) tuples, sorted by preference.
        calculator: ASE calculator for phonon calculations.
        supercell: Supercell dimensions for phonon calculation.
        max_candidates: Maximum number of candidates to test. None = test all.
        log_callback: Optional progress logging callback.

    Returns:
        StabilityTestResult with the first stable candidate found.
    """

    def log(msg: str, level: str = "info"):
        if log_callback:
            log_callback(msg, level)
        if level == "info":
            logger.info(msg)
        elif level == "warn":
            logger.warning(msg)

    if not candidates:
        return StabilityTestResult()

    num_to_test = len(candidates)
    if max_candidates is not None:
        num_to_test = min(num_to_test, max_candidates)

    log(f"Testing up to {num_to_test} candidates for dynamical stability...")

    result = StabilityTestResult()

    for idx in range(num_to_test):
        structure, energy = candidates[idx]
        formula = structure.composition.reduced_formula

        log(
            f"  [{idx + 1}/{num_to_test}] Testing {formula} (E={energy:.4f} eV/atom)..."
        )

        try:
            phonon_result = calculate_phonons(
                structure=structure,
                calculator=calculator,
                supercell=supercell,
                generate_plot=False,
                log_callback=None,  # Suppress detailed logs during batch testing
            )

            result.all_results.append((idx, phonon_result))
            result.num_tested = idx + 1

            if phonon_result.is_stable:
                log(f"  ✓ {formula} is dynamically stable!")
                result.stable_candidate_idx = idx
                result.stable_structure = structure
                result.phonon_result = phonon_result
                return result
            else:
                log(
                    f"  ✗ {formula} is unstable ({phonon_result.num_imaginary_modes} imaginary modes)",
                    level="warn",
                )

        except Exception as e:
            log(f"  ✗ Phonon calculation failed for {formula}: {e}", level="warn")
            # Create a failed result
            failed_result = PhononResult(
                is_stable=False,
                metadata={"error": str(e)},
            )
            result.all_results.append((idx, failed_result))
            result.num_tested = idx + 1

    log(
        f"No dynamically stable candidate found after testing {result.num_tested} structures",
        level="warn",
    )
    return result


# -----------------------------------------------------------------------------
# Integration helpers for ChemistryExplorer
# -----------------------------------------------------------------------------


def test_candidate_stability(
    candidate,  # CandidateResult
    calculator,
    supercell: Tuple[int, int, int] = (2, 2, 2),
    log_callback: Optional[Callable[[str, str], None]] = None,
) -> Tuple[bool, Optional[PhononResult]]:
    """Test a single CandidateResult for dynamical stability.

    Args:
        candidate: CandidateResult with structure to test.
        calculator: ASE calculator for phonon calculations.
        supercell: Supercell dimensions.
        log_callback: Optional progress logging callback.

    Returns:
        Tuple of (is_stable, PhononResult or None if failed).
    """
    structure = candidate.get_structure()
    if structure is None:
        return False, None

    try:
        result = calculate_phonons(
            structure=structure,
            calculator=calculator,
            supercell=supercell,
            generate_plot=False,
            log_callback=log_callback,
        )
        return result.is_stable, result
    except Exception as e:
        logger.warning(f"Phonon calculation failed for {candidate.formula}: {e}")
        return False, None


def select_stable_candidate(
    candidates: List,  # List[CandidateResult]
    calculator,
    supercell: Tuple[int, int, int] = (2, 2, 2),
    max_to_test: Optional[int] = 5,
    log_callback: Optional[Callable[[str, str], None]] = None,
) -> Tuple[Optional[int], Optional[Any], Dict[str, Any]]:
    """Select the best dynamically stable candidate from a list.

    Tests candidates in order until a stable one is found.
    Candidates should be pre-sorted by energy (best first).

    Args:
        candidates: List of CandidateResult objects, sorted by energy.
        calculator: ASE calculator for phonon calculations.
        supercell: Supercell dimensions.
        max_to_test: Maximum candidates to test before giving up.
        log_callback: Optional progress logging callback.

    Returns:
        Tuple of:
        - Index of stable candidate (None if none found)
        - The stable CandidateResult (None if none found)
        - Metadata dict with testing results
    """

    def log(msg: str, level: str = "info"):
        if log_callback:
            log_callback(msg, level)
        logger.log(
            logging.INFO if level == "info" else logging.WARNING,
            msg,
        )

    if not candidates:
        return None, None, {"error": "No candidates provided"}

    valid_candidates = [
        c for c in candidates if c.is_valid and c.get_structure() is not None
    ]
    if not valid_candidates:
        return None, None, {"error": "No valid candidates with structures"}

    # Sort by energy
    valid_candidates.sort(key=lambda c: c.energy_per_atom)

    num_to_test = len(valid_candidates)
    if max_to_test is not None:
        num_to_test = min(num_to_test, max_to_test)

    log(f"Testing {num_to_test} candidates for dynamical stability...")

    tested_results = []
    for idx in range(num_to_test):
        candidate = valid_candidates[idx]
        log(
            f"  [{idx + 1}/{num_to_test}] {candidate.formula} "
            f"(E={candidate.energy_per_atom:.4f} eV/atom, SG={candidate.space_group_symbol})"
        )

        is_stable, phonon_result = test_candidate_stability(
            candidate=candidate,
            calculator=calculator,
            supercell=supercell,
            log_callback=None,  # Suppress detailed logs
        )

        tested_results.append(
            {
                "formula": candidate.formula,
                "energy_per_atom": candidate.energy_per_atom,
                "space_group": candidate.space_group_symbol,
                "is_stable": is_stable,
                "phonon_result": phonon_result,
            }
        )

        if is_stable:
            log(f"  ✓ Found stable candidate: {candidate.formula}")
            return (
                idx,
                candidate,
                {
                    "num_tested": idx + 1,
                    "stable_idx": idx,
                    "tested_results": tested_results,
                },
            )
        else:
            if phonon_result:
                log(
                    f"  ✗ Unstable ({phonon_result.num_imaginary_modes} imaginary modes, "
                    f"min freq: {phonon_result.min_imaginary_frequency:.3f} THz)",
                    level="warn",
                )
            else:
                log("  ✗ Calculation failed", level="warn")

    log(
        f"No stable candidate found after testing {num_to_test} structures",
        level="warn",
    )
    return (
        None,
        None,
        {
            "num_tested": num_to_test,
            "stable_idx": None,
            "tested_results": tested_results,
        },
    )
