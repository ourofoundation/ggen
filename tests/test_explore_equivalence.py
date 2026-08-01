"""Scientific equivalence checks for exploration performance changes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure


def snapshot_exploration(result) -> dict:
    """Return the stable, scientifically relevant portion of an exploration."""
    candidates = []
    for candidate in result.candidates:
        if not candidate.is_valid:
            continue
        structure = candidate.get_structure()
        candidates.append(
            {
                "formula": candidate.formula,
                "energy_per_atom": candidate.energy_per_atom,
                "space_group_number": candidate.space_group_number,
                "structure": structure.as_dict() if structure is not None else None,
            }
        )

    hull_formulas = sorted(candidate.formula for candidate in result.hull_entries)
    return {
        "chemical_system": result.chemical_system,
        "candidates": sorted(candidates, key=lambda item: item["formula"]),
        "hull_formulas": hull_formulas,
    }


def assert_explorations_equivalent(
    baseline: dict,
    actual: dict,
    *,
    energy_atol: float = 1e-4,
) -> None:
    """Assert equivalent structures and energies without comparing CIF bytes."""
    assert actual["chemical_system"] == baseline["chemical_system"]
    assert actual["hull_formulas"] == baseline["hull_formulas"]

    expected_candidates = {
        candidate["formula"]: candidate for candidate in baseline["candidates"]
    }
    actual_candidates = {
        candidate["formula"]: candidate for candidate in actual["candidates"]
    }
    assert actual_candidates.keys() == expected_candidates.keys()

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    for formula, expected in expected_candidates.items():
        observed = actual_candidates[formula]
        assert observed["space_group_number"] == expected["space_group_number"]
        assert observed["energy_per_atom"] == pytest.approx(
            expected["energy_per_atom"], abs=energy_atol
        )
        assert expected["structure"] is not None
        assert observed["structure"] is not None
        assert matcher.fit(
            Structure.from_dict(expected["structure"]),
            Structure.from_dict(observed["structure"]),
        )


@pytest.mark.unit
def test_snapshot_comparison_ignores_site_order_and_float_noise():
    structure = Structure(
        Lattice.cubic(3.5),
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    reordered = Structure(
        Lattice.cubic(3.5),
        ["Fe", "Fe"],
        [[0.5, 0.5, 0.5], [0, 0, 0]],
    )

    def result_with(candidate_structure, energy):
        candidate = SimpleNamespace(
            formula="Fe",
            energy_per_atom=energy,
            space_group_number=229,
            is_valid=True,
            get_structure=lambda: candidate_structure,
        )
        return SimpleNamespace(
            chemical_system="Fe",
            candidates=[candidate],
            hull_entries=[candidate],
        )

    baseline = snapshot_exploration(result_with(structure, -8.0))
    actual = snapshot_exploration(result_with(reordered, -8.00001))
    assert_explorations_equivalent(baseline, actual)


@pytest.mark.integration
@pytest.mark.slow
def test_seeded_fe_bi_exploration_matches_baseline(tmp_path):
    """Run only when a baseline path is supplied on a GPU-capable host."""
    baseline_path_value = os.environ.get("GGEN_EQUIVALENCE_BASELINE")
    if not baseline_path_value:
        pytest.skip("set GGEN_EQUIVALENCE_BASELINE to run the GPU equivalence check")

    from ggen import ChemistryExplorer, StructureDatabase

    baseline_path = Path(baseline_path_value)
    database = StructureDatabase(str(tmp_path / "equivalence.db"))
    explorer = ChemistryExplorer(
        database=database,
        output_dir=tmp_path / "runs",
        random_seed=42,
    )
    result = explorer.explore(
        "Fe-Bi",
        max_atoms=8,
        num_trials=3,
        max_stoichiometries=2,
        optimization_max_steps=100,
        num_workers=1,
        keep_structures_in_memory=True,
        use_unified_database=True,
    )
    actual = snapshot_exploration(result)

    if os.environ.get("GGEN_UPDATE_EQUIVALENCE_BASELINE") == "1":
        baseline_path.write_text(json.dumps(actual, indent=2))
    else:
        baseline = json.loads(baseline_path.read_text())
        assert_explorations_equivalent(baseline, actual)
