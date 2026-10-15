"""SharedDatabase: snapshot readers, working copies, maintenance copies."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from ggen.database import StructureDatabase
from ggen.shared import SharedDatabase


class CountingVolume:
    def __init__(self):
        self.reloads = 0
        self.commits = 0

    def reload(self):
        self.reloads += 1

    def commit(self):
        self.commits += 1


def bcc(element):
    return Structure(Lattice.cubic(2.87), [element, element], [[0, 0, 0], [0.5, 0.5, 0.5]])


def add(db, structure, energy):
    comp = structure.composition
    return db.add_structure(
        formula=comp.formula.replace(" ", ""),
        stoichiometry={el.symbol: int(n) for el, n in comp.items()},
        energy_per_atom=energy,
        total_energy=energy * len(structure),
        num_atoms=len(structure),
        structure=structure,
    )


@pytest.fixture
def shared(tmp_path):
    path = tmp_path / "shared.db"
    db = StructureDatabase(path)
    add(db, bcc("Fe"), -8.0)
    db.close()
    return SharedDatabase(CountingVolume(), path, reload_interval=0)


def test_reload_waits_for_open_readers(shared):
    first = shared.open_reader()
    assert shared.volume.reloads == 1
    second = shared.open_reader()
    # A reader is open, so the view is not swapped underneath it.
    assert shared.volume.reloads == 1
    first.close()
    second.close()
    with shared.reader():
        pass
    assert shared.volume.reloads == 2


def test_working_copy_sends_only_changes(shared):
    sent = []

    def ingest(shard):
        sent.append(shard)
        return shared.merge_shard(shard)

    with shared.working_copy(["Fe-Co"], ingest=ingest) as work:
        add(work, bcc("Co"), -7.0)
    assert len(sent) == 1
    with shared.reader() as db:
        assert {s.formula for s in db.get_structures_for_subsystem("Co-Fe")} == {"Fe2", "Co2"}

    # Nothing changed, nothing sent.
    with shared.working_copy(["Fe"], ingest=ingest):
        pass
    assert len(sent) == 1


def test_exclusive_local_copy_writes_back(shared):
    with shared.exclusive_local_copy() as local:
        db = StructureDatabase(local)
        add(db, bcc("Cr"), -9.0)
        db.close()
    assert shared.volume.commits == 1
    with shared.reader() as db:
        assert db.get_structures_by_formula("Cr2")


def test_failed_maintenance_leaves_file_alone(shared):
    with pytest.raises(RuntimeError):
        with shared.exclusive_local_copy() as local:
            db = StructureDatabase(local)
            add(db, bcc("Cr"), -9.0)
            db.close()
            raise RuntimeError("import failed")
    assert shared.volume.commits == 0
    with shared.reader() as db:
        assert not db.get_structures_by_formula("Cr2")
