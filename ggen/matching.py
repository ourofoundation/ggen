"""Is a structure already known? Structure matching against the reference data.

Everything not generated on Ouro (Materials Project, Alexandria, …) is prior
art. A structure matching one of those is "known"; a structure whose reduced
formula appears there but matches none is a "new polymorph"; a formula absent
from the references is a "new composition". Ouro's own structures (GGen
output and user submissions) are reported separately, since a match there
means "seen on Ouro before", not "published".

Matching uses pymatgen's StructureMatcher on primitive cells with volume
scaling, so the same crystal relaxed by DFT and by an MLIP (a few percent
apart in lattice parameter) still matches, as does a supercell or a
conventional cell of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from pymatgen.core import Structure

if TYPE_CHECKING:
    from .database import StoredStructure, StructureDatabase

logger = logging.getLogger(__name__)

# Sources whose structures were made on Ouro, so they are not prior art.
OURO_SOURCES = ("ggen", "ouro")

KNOWN = "known"
NEW_POLYMORPH = "new polymorph"
NEW_COMPOSITION = "new composition"
REFERENCE = "reference"


@lru_cache(maxsize=1)
def _matcher():
    from pymatgen.analysis.structure_matcher import StructureMatcher

    # Inputs are reduced to primitive cells once, up front, so the matcher
    # does not redo it for every comparison.
    return StructureMatcher(
        ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=False, scale=True
    )


def primitive(structure: Structure) -> Structure:
    try:
        return structure.get_primitive_structure()
    except Exception:
        return structure


def rms_distance(a: Structure, b: Structure) -> Optional[float]:
    """Normalized RMS displacement if two primitive structures match, else None."""
    if len(a) != len(b):
        return None
    try:
        result = _matcher().get_rms_dist(a, b)
    except Exception as exc:
        logger.debug("Structure comparison failed: %s", exc)
        return None
    return None if result is None else float(result[0])


@dataclass
class Match:
    structure_id: str
    source: str
    source_id: Optional[str]
    formula: str
    space_group_number: Optional[int]
    rms_distance: float


@dataclass
class Candidate:
    """A stored structure that shares a reduced formula with the query."""

    structure_id: str
    source: str
    source_id: Optional[str]
    formula: str
    space_group_number: Optional[int]
    cif: str
    _primitive: Optional[Structure] = field(default=None, repr=False)

    def primitive(self) -> Optional[Structure]:
        if self._primitive is None:
            try:
                self._primitive = primitive(Structure.from_str(self.cif, fmt="cif"))
            except Exception as exc:
                logger.debug("Could not parse %s: %s", self.structure_id, exc)
                return None
        return self._primitive


def find_matches(structure: Structure, candidates: Iterable[Candidate]) -> List[Match]:
    """Every candidate that is the same crystal as ``structure``, closest first."""
    query = primitive(structure)
    matches: List[Match] = []
    for candidate in candidates:
        other = candidate.primitive()
        if other is None:
            continue
        rms = rms_distance(query, other)
        if rms is not None:
            matches.append(
                Match(
                    structure_id=candidate.structure_id,
                    source=candidate.source,
                    source_id=candidate.source_id,
                    formula=candidate.formula,
                    space_group_number=candidate.space_group_number,
                    rms_distance=rms,
                )
            )
    return sorted(matches, key=lambda m: m.rms_distance)


@dataclass
class NoveltyResult:
    status: str
    reduced_formula: str
    # Matches in published reference data, closest first.
    matches: List[Match]
    # Matches among Ouro's own structures (GGen output, user submissions).
    ouro_matches: List[Match]
    # Structures with the same reduced formula, by source.
    same_formula_counts: Dict[str, int]


class NoveltyIndex:
    """Classifies structures against the database, one reduced formula at a time.

    Candidates are loaded lazily and cached per formula, so classifying every
    near-hull phase of a system costs one query per distinct formula.
    """

    def __init__(self, db: "StructureDatabase"):
        self.db = db
        self._cache: Dict[str, List[Candidate]] = {}
        self._labels: Dict[str, str] = {}

    def candidates(self, reduced_formula: str) -> List[Candidate]:
        if reduced_formula not in self._cache:
            self._cache[reduced_formula] = self.db.get_match_candidates(reduced_formula)
        return self._cache[reduced_formula]

    def check(
        self,
        structure: Structure,
        exclude_ids: Tuple[str, ...] = (),
        include_ouro: bool = True,
    ) -> NoveltyResult:
        """Match against the references; ``include_ouro`` also matches Ouro's own."""
        reduced = structure.composition.reduced_formula
        candidates = [
            c for c in self.candidates(reduced) if c.structure_id not in exclude_ids
        ]
        counts: Dict[str, int] = {}
        for c in candidates:
            counts[c.source] = counts.get(c.source, 0) + 1

        prior_art = [c for c in candidates if c.source not in OURO_SOURCES]
        ouro = [c for c in candidates if c.source in OURO_SOURCES]
        matches = find_matches(structure, prior_art)
        if matches:
            status = KNOWN
        elif prior_art:
            status = NEW_POLYMORPH
        else:
            status = NEW_COMPOSITION
        return NoveltyResult(
            status=status,
            reduced_formula=reduced,
            matches=matches,
            ouro_matches=find_matches(structure, ouro) if include_ouro else [],
            same_formula_counts=counts,
        )

    def classify(self, stored: "StoredStructure") -> str:
        """Novelty label for a stored structure: reference, known, new polymorph, …"""
        if stored.id not in self._labels:
            self._labels[stored.id] = self._classify(stored)
        return self._labels[stored.id]

    def _classify(self, stored: "StoredStructure") -> str:
        if (stored.source or "ggen") not in OURO_SOURCES:
            return REFERENCE
        structure = stored.get_structure()
        if structure is None:
            # No geometry to match, so fall back to formula and space group.
            prior = [
                c
                for c in self.candidates(stored.reduced_formula)
                if c.source not in OURO_SOURCES
            ]
            if not prior:
                return NEW_COMPOSITION
            known_groups = {c.space_group_number for c in prior}
            return KNOWN if stored.space_group_number in known_groups else NEW_POLYMORPH
        # The label only depends on published structures.
        return self.check(structure, exclude_ids=(stored.id,), include_ouro=False).status
