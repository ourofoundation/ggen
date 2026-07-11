"""Novelty analysis: ggen hull entries vs. Materials Project.

Pure-data core (no torch / orb), so it runs anywhere ``StructureDatabase`` does —
including inside the Modal burn worker. The CLI in ``scripts/discoveries.py`` and
the burn orchestrator both build on these functions.

For each chemical system we recompute the convex hull from all stored structures
(ggen + any imported MP entries) and categorize each ggen hull entry as a
``new_composition`` (no MP structure at that formula), ``beat_mp`` (lower energy
than MP's best), or ``elemental``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from .database import StoredStructure, StructureDatabase


@dataclass
class Discovery:
    """A ggen structure that appears on (or near) the combined hull."""

    structure: StoredStructure
    e_above_hull: float
    category: str  # 'new_composition', 'beat_mp', 'elemental'
    mp_best_energy: Optional[float] = None
    energy_gain: Optional[float] = None  # eV/atom improvement over MP


@dataclass
class SystemDiscoveries:
    """Discovery analysis for one chemical system."""

    chemsys: str
    ggen_on_hull: List[Discovery] = field(default_factory=list)
    mp_on_hull: int = 0
    total_on_hull: int = 0
    ggen_near_hull: List[Discovery] = field(default_factory=list)
    mp_structures_count: int = 0
    ggen_structures_count: int = 0

    @property
    def new_compositions(self) -> List[Discovery]:
        return [d for d in self.ggen_on_hull if d.category == "new_composition"]

    @property
    def beat_mp(self) -> List[Discovery]:
        return [d for d in self.ggen_on_hull if d.category == "beat_mp"]

    @property
    def elementals(self) -> List[Discovery]:
        return [d for d in self.ggen_on_hull if d.category == "elemental"]

    @property
    def non_trivial_discoveries(self) -> List[Discovery]:
        return [d for d in self.ggen_on_hull if d.category != "elemental"]


@dataclass
class UniqueDiscovery:
    """A deduplicated discovery with the systems it appears in."""

    structure_id: str
    formula: str
    space_group: Optional[str]
    energy_per_atom: float
    category: str
    mp_best_energy: Optional[float]
    energy_gain: Optional[float]
    is_dynamically_stable: Optional[bool]
    is_p1: bool
    num_elements: int
    systems: List[str] = field(default_factory=list)


def analyze_system(
    db: StructureDatabase,
    chemsys: str,
    near_hull_cutoff: float = 0.0,
) -> SystemDiscoveries:
    """Analyze discoveries for a single chemical system.

    Computes the hull using all data, then categorizes each ggen hull entry
    based on whether MP had a competing structure at that formula.
    """
    chemsys = db.normalize_chemsys(chemsys)
    result = SystemDiscoveries(chemsys=chemsys)

    best_by_formula = db.get_best_structures_for_subsystem(chemsys)
    if not best_by_formula:
        return result

    mp_best = db.get_best_structures_for_subsystem(chemsys, source="mp")
    ggen_best = db.get_best_structures_for_subsystem(chemsys, source="ggen")
    result.mp_structures_count = len(mp_best)
    result.ggen_structures_count = len(ggen_best)

    entries = []
    entry_structure_pairs: List[Tuple[ComputedEntry, StoredStructure]] = []

    for structure in best_by_formula.values():
        if structure.energy_per_atom is None:
            continue
        comp = Composition(structure.formula)
        energy = structure.energy_per_atom * comp.num_atoms
        entry = ComputedEntry(comp, energy)
        entries.append(entry)
        entry_structure_pairs.append((entry, structure))

    if len(entries) < 2:
        return result

    try:
        pd = PhaseDiagram(entries)
    except Exception:
        return result

    for entry, structure in entry_structure_pairs:
        e_hull = pd.get_e_above_hull(entry)
        is_on_hull = e_hull < 1e-6
        is_near_hull = near_hull_cutoff > 0 and e_hull <= near_hull_cutoff

        if is_on_hull:
            result.total_on_hull += 1

        if structure.source != "ggen":
            if is_on_hull:
                result.mp_on_hull += 1
            continue

        formula_elements = set(Composition(structure.formula).get_el_amt_dict().keys())
        is_elemental = len(formula_elements) == 1

        mp_entry = mp_best.get(structure.formula)
        mp_energy = mp_entry.energy_per_atom if mp_entry else None

        if is_elemental:
            category = "elemental"
        elif mp_energy is None:
            category = "new_composition"
        else:
            category = "beat_mp"

        energy_gain = None
        if mp_energy is not None and structure.energy_per_atom is not None:
            energy_gain = mp_energy - structure.energy_per_atom

        disc = Discovery(
            structure=structure,
            e_above_hull=e_hull,
            category=category,
            mp_best_energy=mp_energy,
            energy_gain=energy_gain,
        )

        if is_on_hull:
            result.ggen_on_hull.append(disc)
        elif is_near_hull:
            result.ggen_near_hull.append(disc)

    result.ggen_on_hull.sort(key=lambda d: d.e_above_hull)
    result.ggen_near_hull.sort(key=lambda d: d.e_above_hull)

    return result


def deduplicate_discoveries(
    all_discoveries: List[SystemDiscoveries],
) -> List[UniqueDiscovery]:
    """Deduplicate discoveries across systems by structure_id.

    Each unique structure is attributed to all the systems it appears in,
    sorted by number of elements (most specific first).
    """
    seen: Dict[str, UniqueDiscovery] = {}

    for sys_disc in all_discoveries:
        for disc in sys_disc.non_trivial_discoveries:
            sid = disc.structure.id
            if sid not in seen:
                s = disc.structure
                formula_elements = set(
                    Composition(s.formula).get_el_amt_dict().keys()
                )
                seen[sid] = UniqueDiscovery(
                    structure_id=sid,
                    formula=s.formula,
                    space_group=s.space_group_symbol,
                    energy_per_atom=s.energy_per_atom,
                    category=disc.category,
                    mp_best_energy=disc.mp_best_energy,
                    energy_gain=disc.energy_gain,
                    is_dynamically_stable=s.is_dynamically_stable,
                    is_p1=s.space_group_symbol == "P1",
                    num_elements=len(formula_elements),
                )
            seen[sid].systems.append(sys_disc.chemsys)

    result = list(seen.values())
    result.sort(key=lambda d: (d.num_elements, d.formula))
    return result


def summarize(
    db: StructureDatabase,
    near_hull_cutoff: float = 0.05,
    top_n: int = 25,
    systems: Optional[List[str]] = None,
) -> Dict:
    """Run the full discovery analysis over the database and return a compact,
    JSON-serializable summary suitable for logging.

    The ``top`` list surfaces the most credible discoveries first:
    phonon-confirmed and higher-symmetry entries before untested / P1 ones.

    Defaults to every multi-element chemical system present in ``structures``
    (not just those with a recorded run, since merged shards have no run rows).
    """
    if systems is None:
        rows = db.conn.execute(
            "SELECT DISTINCT chemsys FROM structures WHERE chemsys LIKE '%-%'"
        ).fetchall()
        systems = [r[0] for r in rows]

    all_discoveries = [
        analyze_system(db, chemsys, near_hull_cutoff=near_hull_cutoff)
        for chemsys in systems
    ]
    unique = deduplicate_discoveries(all_discoveries)

    def rank(d: UniqueDiscovery):
        # Phonon-confirmed first, then higher symmetry, then more elements.
        phonon_rank = 0 if d.is_dynamically_stable is True else (
            1 if d.is_dynamically_stable is None else 2
        )
        return (phonon_rank, d.is_p1, -d.num_elements, d.formula)

    top = sorted(unique, key=rank)[:top_n]

    return {
        "near_hull_cutoff_eV": near_hull_cutoff,
        "systems_analyzed": len(all_discoveries),
        "unique_discoveries": len(unique),
        "new_compositions": sum(1 for d in unique if d.category == "new_composition"),
        "beat_mp": sum(1 for d in unique if d.category == "beat_mp"),
        "phonon_stable": sum(1 for d in unique if d.is_dynamically_stable is True),
        "phonon_unstable": sum(1 for d in unique if d.is_dynamically_stable is False),
        "phonon_untested": sum(1 for d in unique if d.is_dynamically_stable is None),
        "higher_symmetry": sum(1 for d in unique if not d.is_p1),
        "p1": sum(1 for d in unique if d.is_p1),
        "top": [
            {
                "formula": d.formula,
                "space_group": d.space_group,
                "energy_per_atom": d.energy_per_atom,
                "category": d.category,
                "energy_gain_eV": d.energy_gain,
                "is_dynamically_stable": d.is_dynamically_stable,
                "num_elements": d.num_elements,
                "system": min(d.systems, key=len) if d.systems else None,
            }
            for d in top
        ],
    }
