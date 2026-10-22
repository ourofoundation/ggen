"""A run stores each distinct crystal once in the unified database."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

from ggen import StructureDatabase
from ggen.explorer import CandidateResult, ChemistryExplorer, _trial_items


def l10(a=2.55, c=3.6):
    return Structure(Lattice.tetragonal(a, c), ["Cu", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def b2(a=2.88):
    return Structure(Lattice.cubic(a), ["Cu", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def candidate(structure, energy, trials):
    return CandidateResult(
        formula="CuNi",
        stoichiometry={"Cu": 1, "Ni": 1},
        energy_per_atom=energy,
        total_energy=energy * len(structure),
        num_atoms=len(structure),
        space_group_number=123,
        space_group_symbol="P4/mmm",
        structure=structure,
        trials=_trial_items(trials),
    )


def trial(structure, energy):
    return {"structure": structure, "energy_per_atom": energy, "original_space_group": 1, "optimization_steps": 10}


def test_winner_and_trials_store_each_crystal_once(tmp_path):
    db = StructureDatabase(tmp_path / "unified.db")
    explorer = ChemistryExplorer(output_dir=tmp_path / "runs", database=db)
    explorer._unified_run_id = db.create_run("Cu-Ni")
    conn = explorer._init_database(tmp_path)
    # The winner, two more relaxations of it (one in a doubled cell), and a
    # different crystal.
    explorer._save_candidate(
        conn,
        candidate(l10(), -4.2, [trial(l10(), -4.1999), trial(l10() * (1, 1, 2), -4.1998), trial(b2(), -4.0)]),
    )
    rows = db.conn.execute("SELECT id, space_group_symbol FROM structures").fetchall()
    assert len(rows) == 2
    winner = db.conn.execute(
        "SELECT c.energy_per_atom FROM calculations c JOIN structures s ON s.id = c.structure_id "
        "WHERE s.space_group_symbol = 'P4/mmm'"
    ).fetchone()[0]
    assert abs(winner - (-4.2)) < 1e-9
    assert db.conn.execute("SELECT count(*) FROM run_structures").fetchone()[0] == 2

    # A second run that finds the same crystals links them instead.
    explorer._unified_run_id = db.create_run("Cu-Ni")
    explorer._save_candidate(conn, candidate(l10(), -4.21, [trial(b2(), -4.0)]))
    assert db.conn.execute("SELECT count(*) FROM structures").fetchone()[0] == 2
    assert db.conn.execute("SELECT count(*) FROM run_structures").fetchone()[0] == 4
    db.close()


def test_a_winner_that_loses_to_a_known_structure_is_still_stored(tmp_path):
    db = StructureDatabase(tmp_path / "unified.db")
    explorer = ChemistryExplorer(output_dir=tmp_path / "runs", database=db)
    explorer._unified_run_id = db.create_run("Cu-Ni")
    conn = explorer._init_database(tmp_path)
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    # The database already holds a lower B2 cell for CuNi.
    known = candidate(b2(), -4.3, [])
    known.generation_metadata["loaded_from_unified_db"] = True
    ours = candidate(l10(), -4.1, [trial(b2(3.1), -4.0)])

    recorded, ok = explorer._record_candidate(
        ours, "CuNi", {"CuNi": known}, structures_dir, conn, False, (2, 2, 2), False, True
    )
    assert ok and recorded is known
    # Our winner and its trial are both this run's structures.
    symbols = sorted(
        row[0]
        for row in db.conn.execute(
            "SELECT s.space_group_symbol FROM run_structures rs "
            "JOIN structures s ON s.id = rs.structure_id"
        )
    )
    assert symbols == ["P4/mmm", "Pm-3m"]
    db.close()
