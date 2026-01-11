from typing import Literal, Tuple

import numpy as np


def compute_fmax(
    forces: np.ndarray,
    mode: Literal["norm", "component"] = "norm",
) -> float:
    """Compute the maximum force from a forces array.

    Args:
        forces: Array of shape (n_atoms, 3) containing force vectors.
        mode: How to compute fmax:
            - "norm": Max magnitude of force vectors (sqrt of sum of squares).
              This is what ASE optimizers use for convergence.
            - "component": Max absolute value of any force component.
              Simpler but stricter metric.

    Returns:
        Maximum force value.
    """
    if mode == "norm":
        # Max vector magnitude (ASE optimizer style)
        return float(np.sqrt((forces**2).sum(axis=1).max()))
    elif mode == "component":
        # Max absolute component
        return float(np.max(np.abs(forces)))
    else:
        raise ValueError(f"Unknown fmax mode: {mode}")


def parse_chemical_formula(formula: str) -> Tuple[list[str], list[int]]:
    """Parse a chemical formula string into elements and counts using pymatgen.

    Args:
        formula: Chemical formula string, e.g. 'SiO2', 'Fe2O3', 'BaTiO3'

    Returns:
        Tuple of (elements, counts) where elements is a list of element symbols
        and counts is a list of corresponding stoichiometric coefficients.

    Examples:
        >>> parse_chemical_formula('SiO2')
        (['Si', 'O'], [1, 2])
        >>> parse_chemical_formula('BaTiO3')
        (['Ba', 'Ti', 'O'], [1, 1, 3])
    """
    from pymatgen.core import Composition

    # Use pymatgen's Composition class for robust formula parsing
    composition = Composition(formula)

    # Get elements and their counts
    elements = [str(element) for element in composition.elements]
    counts = [int(composition[element]) for element in composition.elements]

    return elements, counts
