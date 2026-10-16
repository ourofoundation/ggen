"""MP2020 energy corrections, applied the same way to every surface.

Materials Project and Alexandria both compute PBE with +U on transition
metals in oxides and fluorides, but they publish differently corrected
energies (Alexandria's ``energy_corrected`` is off MP's by a median 42
meV/atom on matched structures, while their raw energies agree to ~8). So we
store raw energies and apply MaterialsProject2020Compatibility ourselves.

MLIPs trained on MPtrj / sAlex reproduce that GGA/GGA+U mix, so their raw
energies need the same treatment before they can share a hull. For those,
and for Alexandria, the run type follows MP's input-set rule:
``hubbard_parameters`` picks GGA+U exactly when MP would (it agreed with MP
on all 918 entries of a mixed oxide/fluoride/intermetallic sample).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _compatibility():
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

    return MaterialsProject2020Compatibility(check_potcar=False)


@lru_cache(maxsize=1)
def _u_settings() -> Dict[str, Dict[str, float]]:
    """U values by anion ("O", "F") then metal, from pymatgen's MP2020 config."""
    return {
        anion: {el: float(u) for el, u in values.items()}
        for anion, values in _compatibility().u_settings.items()
    }


def hubbard_parameters(composition: Composition) -> Dict[str, Any]:
    """Run parameters MP would have used for a composition."""
    elements = {el.symbol for el in Composition(composition).elements}
    for anion in ("F", "O"):
        u_values = _u_settings().get(anion, {})
        if anion in elements and elements & u_values.keys():
            return {
                "run_type": "GGA+U",
                "is_hubbard": True,
                "hubbards": {el: u_values.get(el, 0.0) for el in sorted(elements)},
            }
    return {"run_type": "GGA", "is_hubbard": False, "hubbards": {}}


@lru_cache(maxsize=1)
def _corrected_elements() -> frozenset:
    """Elements MP2020 can adjust for; oxide types all mean oxygen."""
    keys = set(_compatibility().comp_correction)
    oxide_types = {"oxide", "peroxide", "superoxide", "ozonide"}
    return frozenset((keys - oxide_types) | ({"O"} if keys & oxide_types else set()))


@dataclass
class Correction:
    """MP2020 result for one entry. ``per_atom`` is None if MP2020 rejects it."""

    per_atom: Optional[float]
    run_type: str
    hubbards: Dict[str, float]

    @property
    def is_compatible(self) -> bool:
        return self.per_atom is not None


def mp2020_correction(
    structure: Structure,
    energy_per_atom: float,
    parameters: Optional[Dict[str, Any]] = None,
) -> Correction:
    """MP2020 adjustment per atom for a raw energy on ``structure``.

    ``parameters`` are the calculation's own run settings when known (a
    Materials Project entry); otherwise MP's rule is applied. The energy's cell
    may differ from ``structure``'s (Alexandria stores a primitive-cell energy
    beside a conventional CIF), which is why everything here is per atom.
    """
    params = dict(parameters) if parameters else hubbard_parameters(structure.composition)
    # Without any element MP2020 adjusts (and so without +U, which needs O or
    # F) there is nothing to correct; skip pymatgen's oxidation-state work,
    # most of an import's time.
    elements = {el.symbol for el in structure.composition.elements}
    if not elements & _corrected_elements() and params.get("run_type") != "GGA+U":
        return Correction(per_atom=0.0, run_type=params.get("run_type", "GGA"), hubbards={})
    entry = ComputedStructureEntry(
        structure, energy_per_atom * len(structure), parameters=params
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        processed = _compatibility().process_entry(entry)
    per_atom = None
    if processed is not None:
        per_atom = processed.correction / len(structure)
    return Correction(
        per_atom=per_atom,
        run_type=params.get("run_type", "GGA"),
        hubbards=dict(params.get("hubbards") or {}),
    )
