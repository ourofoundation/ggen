"""Structure database: surfaces, corrected hulls, shards and novelty."""

from __future__ import annotations

import sqlite3

import pytest
from pymatgen.core import Lattice, Structure

from ggen.database import StructureDatabase
from ggen.matching import KNOWN, NEW_COMPOSITION, NEW_POLYMORPH, REFERENCE
from ggen.surfaces import ORB_V3, PBE, SOURCE_GEOMETRY


def bcc(element: str, a: float = 2.87) -> Structure:
    return Structure(Lattice.cubic(a), [element, element], [[0, 0, 0], [0.5, 0.5, 0.5]])


def fcc(element: str, a: float = 3.6) -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [element], [[0, 0, 0]])


def b2(a: str, b: str, lattice: float = 2.9) -> Structure:
    return Structure(Lattice.cubic(lattice), [a, b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def rocksalt(cation: str, anion: str, a: float = 4.3) -> Structure:
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def add(db: StructureDatabase, structure: Structure, energy, source="ggen", **kwargs):
    comp = structure.composition
    return db.add_structure(
        formula=comp.formula.replace(" ", ""),
        stoichiometry={el.symbol: int(n) for el, n in comp.items()},
        energy_per_atom=energy,
        total_energy=None if energy is None else energy * len(structure),
        num_atoms=len(structure),
        structure=structure,
        source=source,
        **kwargs,
    )


@pytest.fixture
def db(tmp_path):
    database = StructureDatabase(tmp_path / "ggen.db")
    yield database
    database.close()


def test_hull_uses_lowest_corrected_energy_per_formula(db):
    fe = add(db, bcc("Fe"), -8.0)
    ni = add(db, fcc("Ni"), -5.5)
    stable = add(db, b2("Fe", "Ni"), -7.0)
    polymorph = add(db, b2("Fe", "Ni", lattice=3.0), -6.95)
    above = add(db, b2("Fe", "Ni", lattice=3.1), -6.5)

    pd, e_hull, _ = db.compute_hull("Ni-Fe")
    assert pd is not None
    assert e_hull[fe] == pytest.approx(0.0)
    assert e_hull[stable] == pytest.approx(0.0)
    # Polymorphs get hull distances even though only the best is a phase.
    assert e_hull[polymorph] == pytest.approx(0.05)
    assert e_hull[above] == pytest.approx(0.5)

    near = db.get_hull_entries("Fe-Ni", e_above_hull_cutoff=0.1)
    assert {s.id for s in near} == {fe, ni, stable}
    assert db.get_structure(polymorph).e_above_hull == pytest.approx(0.05)


def test_mp2020_corrections_apply_to_ml_energies(db):
    """FeO gets the Fe +U mixing and oxide corrections; metallic Fe gets none."""
    fe = add(db, bcc("Fe"), -8.3)
    feo = add(db, rocksalt("Fe", "O"), -7.0)
    fe_stored, feo_stored = db.get_structure(fe), db.get_structure(feo)
    assert fe_stored.correction_per_atom == pytest.approx(0.0)
    assert feo_stored.correction_per_atom < -1.0
    row = db.conn.execute(
        "SELECT run_type, hubbards FROM calculations WHERE structure_id = ?", (feo,)
    ).fetchone()
    assert row["run_type"] == "GGA+U"
    assert '"Fe": 5.3' in row["hubbards"]


def test_surfaces_are_independent(tmp_path):
    path = tmp_path / "ggen.db"
    orb = StructureDatabase(path)
    fe = add(orb, bcc("Fe"), -8.0)
    ni = add(orb, fcc("Ni"), -5.5)
    alloy = add(orb, b2("Fe", "Ni"), -7.0)
    orb.compute_hull("Fe-Ni")

    pbe = StructureDatabase(path, surface=PBE.id)
    # No PBE energies yet: nothing to build a hull from.
    assert pbe.compute_hull("Fe-Ni")[0] is None
    for sid, energy in ((fe, -8.2), (ni, -5.6), (alloy, -6.8)):
        pbe.set_calculation(sid, energy)
    pbe.compute_hull("Fe-Ni")
    assert pbe.get_structure(alloy).e_above_hull == pytest.approx(0.1)
    # ...and the ORB hull is untouched.
    assert orb.get_structure(alloy).e_above_hull == pytest.approx(0.0)
    assert orb.get_structure(alloy).energy_per_atom == pytest.approx(-7.0)
    orb.close()
    pbe.close()


def test_source_correction_is_kept(db):
    sid = add(db, rocksalt("Fe", "O"), None, source="mp", source_id="mp-1")
    db.set_calculation(sid, -7.0, surface=PBE.id, correction_per_atom=-1.234)
    row = db.conn.execute(
        "SELECT correction_per_atom FROM calculations WHERE structure_id = ? AND surface = ?",
        (sid, PBE.id),
    ).fetchone()
    assert row[0] == pytest.approx(-1.234)


def test_geometry_slots(db):
    sid = add(db, bcc("Fe"), None, source="mp", source_id="mp-13")
    assert db.get_geometry(sid, SOURCE_GEOMETRY) is not None
    assert db.get_geometry(sid, ORB_V3.id) is None
    relaxed = bcc("Fe", a=2.83).to(fmt="cif")
    db._update_structure(sid, energy_per_atom=-8.1, total_energy=-16.2, cif_content=relaxed)
    assert db.get_geometry(sid, ORB_V3.id) == relaxed
    # The surface's own geometry wins over the published one.
    assert "2.83" in db.get_structure(sid).cif_content


def test_export_and_merge_round_trip(tmp_path):
    main = StructureDatabase(tmp_path / "main.db")
    fe = add(main, bcc("Fe"), -8.0)
    add(main, fcc("Ni"), -5.5)
    add(main, rocksalt("Fe", "O"), None, source="mp", source_id="mp-1")

    # A worker copies the system, adds structures, and exports what changed.
    main.extract_systems(tmp_path / "work.db", ["Fe-Ni"])
    work = StructureDatabase(tmp_path / "work.db")
    assert {s.formula for s in work.get_structures_for_subsystem("Fe-Ni")} == {"Fe2", "Ni4"}
    from datetime import datetime

    since = datetime.now().isoformat()
    run = work.create_run("Fe-Ni")
    new = add(work, b2("Fe", "Ni"), -7.0, run_id=run)
    work._update_structure(fe, is_dynamically_stable=True)
    assert work.export_changes(tmp_path / "delta.db", since) == 2
    work.close()

    imported, merged = main.merge_from(tmp_path / "delta.db")
    assert (imported, merged) == (1, 1)
    assert main.get_structure(new).e_above_hull == pytest.approx(0.0)
    assert main.get_structure(fe).is_dynamically_stable is True
    assert main.get_run_statistics(run)["structures"] == 1

    # Re-importing a source structure under a new id lands on the existing row.
    other = StructureDatabase(tmp_path / "other.db")
    add(other, rocksalt("Fe", "O"), None, source="mp", source_id="mp-1")
    other.close()
    assert main.merge_from(tmp_path / "other.db", recompute_hulls=False) == (0, 1)
    assert main.conn.execute(
        "SELECT COUNT(*) FROM structures WHERE source_id = 'mp-1'"
    ).fetchone()[0] == 1
    main.close()


def test_readonly_extract(tmp_path):
    path = tmp_path / "ggen.db"
    writer = StructureDatabase(path)
    add(writer, bcc("Fe"), -8.0)
    writer.close()

    reader = StructureDatabase(path, readonly=True, snapshot=True)
    with pytest.raises(sqlite3.OperationalError):
        reader.conn.execute("DELETE FROM structures")
    assert reader.extract_systems(tmp_path / "copy.db", ["Fe"]) == 1
    with pytest.raises(sqlite3.OperationalError):
        reader.conn.execute("DELETE FROM structures")
    reader.close()


def test_novelty(db):
    add(db, b2("Fe", "Ni"), None, source="mp", source_id="mp-2")
    index = db.novelty_index()

    # Same crystal at a different volume (DFT vs MLIP lattice) and as a supercell.
    assert index.check(b2("Fe", "Ni", lattice=2.95)).status == KNOWN
    supercell = b2("Fe", "Ni")
    supercell.make_supercell([2, 1, 1])
    result = index.check(supercell)
    assert result.status == KNOWN
    assert result.matches[0].source_id == "mp-2"

    # Same formula, different crystal: L1_0-like tetragonal cell.
    tetragonal = Structure(
        Lattice.tetragonal(2.5, 3.6), ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    assert index.check(tetragonal).status == NEW_POLYMORPH
    assert index.check(b2("Fe", "Co")).status == NEW_COMPOSITION

    # GGen's own structures are reported separately, not as prior art.
    ggen_id = add(db, b2("Fe", "Co"), -7.0)
    index = db.novelty_index()
    result = index.check(b2("Fe", "Co"))
    assert result.status == NEW_COMPOSITION
    assert [m.structure_id for m in result.ouro_matches] == [ggen_id]
    assert index.classify(db.get_structure(ggen_id)) == NEW_COMPOSITION
    mp = db.get_structure_by_source("mp", "mp-2")
    assert index.classify(mp) == REFERENCE


def test_refuses_version_1_schema(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE structures (id TEXT PRIMARY KEY, cif_content TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="version 1"):
        StructureDatabase(path)
