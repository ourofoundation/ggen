"""
GGen: Crystal Generation and Mutation Library

A powerful Python library for crystal structure generation, mutation, and evolutionary optimization.
Built on top of PyXtal, pymatgen, and ASE, GGen provides an intuitive interface for generating,
modifying, and analyzing crystal structures with built-in energy evaluation using ORB models.
"""

from .calculator import get_orb_calculator
from .colors import Colors
from .database import ExplorationRun, StoredStructure, StructureDatabase
from .explorer import CandidateResult, ChemistryExplorer, ExplorationResult
from .ggen import GGen
from .operations import MutationError, Operations
from .phonons import (
    PhononResult,
    StabilityTestResult,
    calculate_phonons,
    check_dynamical_stability,
    find_first_stable_candidate,
    select_stable_candidate,
)
from .utils import parse_chemical_formula

__version__ = "0.1.0"
__author__ = "Matt Moderwell"
__email__ = "matt@ouro.foundation"

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
    # Utilities
    "Colors",
    "get_orb_calculator",
    "parse_chemical_formula",
]
