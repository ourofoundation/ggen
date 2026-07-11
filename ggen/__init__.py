"""
GGen: Crystal Generation and Mutation Library

A powerful Python library for crystal structure generation, mutation, and evolutionary optimization.
Built on top of PyXtal, pymatgen, and ASE, GGen provides an intuitive interface for generating,
modifying, and analyzing crystal structures with built-in energy evaluation using ORB models.

Heavy symbols that pull torch / orb / torch-sim / phonopy at import time are
lazy-loaded via PEP 562 ``__getattr__``, so callers that only need the
lightweight database / report / operations APIs (e.g. read-only consumers in a
serverless web tier) don't pay the cost of the GPU stack just to do
``from ggen import StructureDatabase``.
"""

from .colors import Colors
from .database import ExplorationRun, StoredStructure, StructureDatabase
from .elements import get_element_group, list_groups, resolve_candidates
from .operations import MutationError, Operations
from .report import SpaceGroupStats, StabilityStats, SystemExplorer, SystemReport
from .utils import parse_chemical_formula

__version__ = "0.1.0"
__author__ = "Matt Moderwell"
__email__ = "matt@ouro.foundation"


# Lazy imports: each entry maps an attribute name -> (submodule, attr_in_submodule).
# Importing the submodule pulls in torch / orb / torch-sim / phonopy, so we defer
# until first attribute access on the package.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # ggen.ggen — torch + torch-sim + ORB calculator
    "GGen": (".ggen", "GGen"),
    "get_space_group_cache_info": (".ggen", "get_space_group_cache_info"),
    "clear_space_group_cache": (".ggen", "clear_space_group_cache"),
    # ggen.calculator — torch + orb_models
    "get_orb_calculator": (".calculator", "get_orb_calculator"),
    # ggen.explorer — uses calculator + torch
    "ChemistryExplorer": (".explorer", "ChemistryExplorer"),
    "CandidateResult": (".explorer", "CandidateResult"),
    "ExplorationResult": (".explorer", "ExplorationResult"),
    # ggen.scout — uses explorer + torch
    "SystemScout": (".scout", "SystemScout"),
    "SystemScore": (".scout", "SystemScore"),
    "ScoutResult": (".scout", "ScoutResult"),
    # ggen.phonons — phonopy
    "PhononResult": (".phonons", "PhononResult"),
    "StabilityTestResult": (".phonons", "StabilityTestResult"),
    "calculate_phonons": (".phonons", "calculate_phonons"),
    "check_dynamical_stability": (".phonons", "check_dynamical_stability"),
    "find_first_stable_candidate": (".phonons", "find_first_stable_candidate"),
    "select_stable_candidate": (".phonons", "select_stable_candidate"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute loader for heavy submodules."""
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name, package=__name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys())))


__all__ = [
    # Core
    "GGen",
    # Explorer
    "ChemistryExplorer",
    "CandidateResult",
    "ExplorationResult",
    # Database
    "StructureDatabase",
    "StoredStructure",
    "ExplorationRun",
    # Reporting / Analysis
    "SystemExplorer",
    "SystemReport",
    "StabilityStats",
    "SpaceGroupStats",
    # Operations
    "Operations",
    "MutationError",
    # Phonons / Dynamical Stability
    "PhononResult",
    "StabilityTestResult",
    "calculate_phonons",
    "check_dynamical_stability",
    "find_first_stable_candidate",
    "select_stable_candidate",
    # Scout
    "SystemScout",
    "SystemScore",
    "ScoutResult",
    # Elements
    "get_element_group",
    "list_groups",
    "resolve_candidates",
    # Utilities
    "Colors",
    "get_orb_calculator",
    "parse_chemical_formula",
    # Cache management
    "get_space_group_cache_info",
    "clear_space_group_cache",
]
