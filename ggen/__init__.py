"""
GGen: Crystal Generation and Mutation Library

A powerful Python library for crystal structure generation, mutation, and evolutionary optimization.
Built on top of PyXtal, pymatgen, and ASE, GGen provides an intuitive interface for generating,
modifying, and analyzing crystal structures with built-in energy evaluation using ORB models.
"""

from .calculator import get_orb_calculator
from .explorer import CandidateResult, ChemistryExplorer, ExplorationResult
from .ggen import GGen
from .operations import MutationError, Operations
from .utils import parse_chemical_formula

__version__ = "0.1.0"
__author__ = "Matt Moderwell"
__email__ = "matt@ouro.foundation"

__all__ = [
    "GGen",
    "ChemistryExplorer",
    "CandidateResult",
    "ExplorationResult",
    "Operations",
    "MutationError",
    "get_orb_calculator",
    "parse_chemical_formula",
]
