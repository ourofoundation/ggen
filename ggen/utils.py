from typing import Tuple


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
