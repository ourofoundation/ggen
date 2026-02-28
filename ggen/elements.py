"""Element group definitions for systematic chemical space exploration.

Provides named groups of elements (e.g., '3d_metals', 'chalcogens') and
utilities for resolving candidate element lists from group names with
exclusion support.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pymatgen.core.periodic_table import Element

# Maximum atomic number to include (excludes radioactive elements beyond Bi)
MAX_ATOMIC_NUMBER = 83

ELEMENT_GROUPS: Dict[str, List[str]] = {
    "3d_metals": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    "4d_metals": ["Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag"],
    "5d_metals": ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"],
    "post_transition_metals": ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi"],
    "alkaline_earth": ["Mg", "Ca", "Sr", "Ba"],
    "alkali": ["Li", "Na", "K", "Rb", "Cs"],
    "metalloids": ["B", "Si", "Ge", "As", "Sb", "Te"],
    "chalcogens": ["O", "S", "Se", "Te"],
    "pnictogens": ["N", "P", "As", "Sb", "Bi"],
    "rare_earth": [
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    ],
}

# Composite groups built from the above
ELEMENT_GROUPS["transition_metals"] = (
    ELEMENT_GROUPS["3d_metals"]
    + ELEMENT_GROUPS["4d_metals"]
    + ELEMENT_GROUPS["5d_metals"]
)
ELEMENT_GROUPS["metals"] = (
    ELEMENT_GROUPS["transition_metals"]
    + ELEMENT_GROUPS["post_transition_metals"]
    + ELEMENT_GROUPS["alkaline_earth"]
    + ELEMENT_GROUPS["alkali"]
)
ELEMENT_GROUPS["nonmetals"] = (
    ELEMENT_GROUPS["chalcogens"]
    + ELEMENT_GROUPS["pnictogens"]
    + ["H", "C", "F", "Cl", "Br", "I"]
)


def get_element_group(name: str) -> List[str]:
    """Get elements in a named group.

    Args:
        name: Group name (e.g., '3d_metals', 'transition_metals').

    Returns:
        List of element symbols.

    Raises:
        ValueError: If group name is not recognized.
    """
    if name not in ELEMENT_GROUPS:
        available = ", ".join(sorted(ELEMENT_GROUPS.keys()))
        raise ValueError(f"Unknown element group '{name}'. Available: {available}")
    return list(ELEMENT_GROUPS[name])


def list_groups() -> Dict[str, List[str]]:
    """Return all available element groups."""
    return {k: list(v) for k, v in ELEMENT_GROUPS.items()}


def resolve_candidates(
    groups: Optional[List[str]] = None,
    elements: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    max_z: int = MAX_ATOMIC_NUMBER,
) -> List[str]:
    """Resolve a list of candidate elements from groups and explicit lists.

    Combines elements from named groups and/or an explicit element list,
    applies exclusions, and filters out noble gases and radioactive elements.

    Args:
        groups: Named element groups to include (e.g., ['3d_metals']).
        elements: Explicit element symbols to include.
        exclude: Element symbols to exclude from the result.
        max_z: Maximum atomic number to include (default: 83, excludes
            radioactive elements beyond Bi).

    Returns:
        Sorted list of unique element symbols.

    Raises:
        ValueError: If neither groups nor elements is provided, or if
            a group name is not recognized.
    """
    if not groups and not elements:
        raise ValueError("Must provide at least one of 'groups' or 'elements'")

    candidates: set[str] = set()

    if groups:
        for group_name in groups:
            candidates.update(get_element_group(group_name))

    if elements:
        for el in elements:
            Element(el)  # validate
            candidates.add(el)

    exclude_set = set(exclude) if exclude else set()

    # Auto-exclude noble gases
    noble_gases = {"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"}
    exclude_set.update(noble_gases)

    result = []
    for el in sorted(candidates):
        if el in exclude_set:
            continue
        if Element(el).Z > max_z:
            continue
        result.append(el)

    return result
