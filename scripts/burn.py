"""Parallel structure-search burn for ggen on Modal.

Drains a GPU credit budget by fanning a systematic ternary sweep (Campaign B)
and a phonon + novelty backfill (Campaign C) across many Modal containers,
writing into one shared ggen database without ever letting two containers
write that database at once.

Design
------
* ``explore_one`` runs on GPU containers and is the only thing that scales out.
  Each call works on a private ``/tmp`` copy of a *seed snapshot* (the lower-arity
  results), explores one chemical system, and returns a small **shard** holding
  only the structures it newly generated. It never touches the master DB.
* ``MasterDB`` is a singleton (``max_containers=1``) that owns the master
  ``ggen.db`` on the volume. The local driver calls it serially to merge shards,
  snapshot seeds, track spend, and run phonon updates — so writes never race.
* The driver (``main``) is idempotent: it skips systems already in the master DB,
  so a killed run resumes by rerunning the same command.

Staging eliminates redundant compute: unaries first, then binaries seeded with
the unaries, then ternaries seeded with binaries+unaries. With ``require_all_elements``
each phase only generates its own arity; endpoints come from the seed.

Usage
-----
    # See the plan and cost estimate without launching anything
    modal run scripts/burn.py --dry-run

    # Full B + C burn (defaults: 20-element palette, A10G, ~$3.7k cap)
    modal run scripts/burn.py

    # Smoke test on a tiny palette with low concurrency first
    GGEN_BURN_MAX_CONTAINERS=4 modal run scripts/burn.py --palette Fe,Co,Si,Mn,Ni

    # Bigger/faster GPUs to drain credits quicker
    GGEN_BURN_GPU=A100 modal run scripts/burn.py

    # Only the phonon/novelty backfill on whatever is already in the DB
    modal run scripts/burn.py --stage phonon
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import modal

GGEN_ROOT = Path(__file__).parent.parent

# GPU type and max parallel containers are fixed at deploy/run time (Modal 1.2
# has no per-call ``with_options``), so they're controlled by env vars read when
# this module is imported by ``modal run``:
#     GGEN_BURN_GPU=A10G  GGEN_BURN_MAX_CONTAINERS=15  modal run scripts/burn.py
BURN_GPU = os.environ.get("GGEN_BURN_GPU", "A10G")
BURN_MAX_CONTAINERS = int(os.environ.get("GGEN_BURN_MAX_CONTAINERS", "15"))

# 20-element palette: synthesizable, MLIP-friendly (no noble gases / radioactives
# / f-block). C(20,3) = 1140 ternaries, C(20,2) = 190 binaries.
DEFAULT_PALETTE = [
    "Li", "Mg", "Al", "Si", "P", "S", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "Se", "Sn", "Sb",
]

# Modal list prices ($/GPU-hour), used only for the spend estimate / guard.
GPU_HOURLY_RATE = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}

VOL_MOUNT = "/vol"
# Same file the deployed materials/apps/ggen app uses: volume ``ggen-data``,
# path ``data/ggen.db`` inside it.
MASTER_DB = f"{VOL_MOUNT}/data/ggen.db"
BURN_DIR = f"{VOL_MOUNT}/burn"
CACHE_DIR = f"{VOL_MOUNT}/.cache"

volume = modal.Volume.from_name("ggen-data", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pymatgen>=2023.0.0",
        "pyxtal>=0.5.0",
        "ase>=3.22.0",
        "orb-models>=0.6.0",
        "nvalchemi-toolkit-ops>=0.2.0",
        "pynanoflann",
        "torch-sim-atomistic>=0.5.2",
        "cuml-cu12==25.2.*",
        "phonopy>=2.20.0",
        "seekpath>=2.0",
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "matplotlib>=3.0",
    )
    .add_local_dir(GGEN_ROOT / "ggen", remote_path="/root/ggen_pkg/ggen")
)

app = modal.App(name="ggen-burn", image=image)


# ---------------------------------------------------------------------------
# Helpers shared by the GPU workers.
# ---------------------------------------------------------------------------
def _prepare_runtime() -> None:
    """Point model caches at the volume and put the mounted ggen pkg on the path."""
    import os
    import sys

    os.environ.setdefault("CACHED_PATH_CACHE_ROOT", f"{CACHE_DIR}/cached_path")
    os.environ.setdefault("HF_HOME", f"{CACHE_DIR}/huggingface")
    os.environ.setdefault("TORCH_HOME", f"{CACHE_DIR}/torch")
    for sub in ("cached_path", "huggingface", "torch"):
        os.makedirs(f"{CACHE_DIR}/{sub}", exist_ok=True)
    sys.path.insert(0, "/root/ggen_pkg")


def _adaptive_supercell(num_atoms: int, min_atoms: int = 150, max_dim: int = 5):
    for n in range(3, max_dim + 1):
        if num_atoms * (n**3) >= min_atoms:
            return (n, n, n)
    return (max_dim, max_dim, max_dim)


# ---------------------------------------------------------------------------
# GPU worker: explore one chemical system, return a shard of only-new structures.
# ---------------------------------------------------------------------------
@app.function(
    gpu=BURN_GPU,
    cpu=2,
    timeout=7200,
    max_containers=BURN_MAX_CONTAINERS,
    retries=1,
    volumes={VOL_MOUNT: volume},
)
def explore_one(system: str, params: dict, seed_rel: str | None) -> dict:
    import shutil
    import time
    import warnings

    _prepare_runtime()
    from ggen import ChemistryExplorer, StructureDatabase

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    work = "/tmp/work.db"
    for stale in (work, f"{work}-wal", f"{work}-shm"):
        Path(stale).unlink(missing_ok=True)

    if seed_rel:
        volume.reload()
        seed_path = Path(BURN_DIR) / seed_rel
        if seed_path.exists():
            shutil.copy(seed_path, work)

    db = StructureDatabase(work)
    pre_ids = {row[0] for row in db.conn.execute("SELECT id FROM structures")}

    explorer = ChemistryExplorer(output_dir="/tmp/runs", database=db)
    t0 = time.time()
    try:
        explorer.explore(
            chemical_system=system,
            max_atoms=params["max_atoms"],
            min_atoms=params["min_atoms"],
            num_trials=params["num_trials"],
            optimize=True,
            include_binaries=True,
            include_ternaries=True,
            require_all_elements=params["require_all_elements"],
            max_stoichiometries=params["max_stoichiometries"],
            skip_existing_formulas=True,
            preserve_symmetry=False,
            num_workers=1,
            show_progress=False,
            keep_structures_in_memory=False,
            use_unified_database=True,
            compute_phonons=False,
            optimization_max_steps=params["max_steps"],
            optimization_optimizer="fire",
        )
    except Exception as exc:
        db.close()
        return {"system": system, "error": str(exc), "elapsed_s": time.time() - t0}

    elapsed = time.time() - t0
    post_ids = {row[0] for row in db.conn.execute("SELECT id FROM structures")}
    new_ids = list(post_ids - pre_ids)

    shard_path = "/tmp/shard.db"
    Path(shard_path).unlink(missing_ok=True)
    new_count = db.extract_subset(shard_path, new_ids)
    db.close()

    shard_bytes = Path(shard_path).read_bytes() if new_count else b""
    return {
        "system": system,
        "error": None,
        "elapsed_s": elapsed,
        "new_count": new_count,
        "shard_bytes": shard_bytes,
    }


# ---------------------------------------------------------------------------
# GPU worker: phonons for a batch of structures (no master DB access).
# ---------------------------------------------------------------------------
@app.function(
    gpu=BURN_GPU,
    cpu=2,
    timeout=7200,
    max_containers=BURN_MAX_CONTAINERS,
    volumes={VOL_MOUNT: volume},
)
def phonons_batch(items: list[dict]) -> list[dict]:
    import time
    import warnings

    _prepare_runtime()
    from pymatgen.core import Structure

    from ggen.calculator import get_orb_calculator
    from ggen.phonons import calculate_phonons

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    calculator = get_orb_calculator()
    results = []
    for item in items:
        out = {"id": item["id"], "elapsed_s": 0.0}
        t0 = time.time()
        try:
            structure = Structure.from_str(item["cif"], fmt="cif")
            supercell = _adaptive_supercell(item["num_atoms"])
            res = calculate_phonons(
                structure=structure,
                calculator=calculator,
                supercell=supercell,
                generate_plot=False,
            )
            out.update(
                {
                    "is_stable": bool(res.is_stable),
                    "num_imaginary_modes": int(res.num_imaginary_modes),
                    "min_frequency": float(res.min_frequency),
                    "max_frequency": float(res.max_frequency),
                    "supercell": list(supercell),
                }
            )
        except Exception as exc:
            out["error"] = str(exc)
        out["elapsed_s"] = time.time() - t0
        results.append(out)
    return results


# ---------------------------------------------------------------------------
# Singleton owner of the master DB. All writes funnel through here, serially.
# ---------------------------------------------------------------------------
@app.cls(cpu=2, timeout=3600, volumes={VOL_MOUNT: volume}, max_containers=1)
class MasterDB:
    @modal.enter()
    def setup(self):
        _prepare_runtime()
        from ggen import StructureDatabase

        Path(MASTER_DB).parent.mkdir(parents=True, exist_ok=True)
        Path(BURN_DIR).mkdir(parents=True, exist_ok=True)
        self.db = StructureDatabase(MASTER_DB)

    def _persist(self):
        self.db.conn.commit()
        volume.commit()

    @modal.method()
    def done_systems(self) -> list[str]:
        rows = self.db.conn.execute("SELECT DISTINCT chemsys FROM structures")
        return [r[0] for r in rows]

    @modal.method()
    def merge_shard(self, shard_bytes: bytes) -> dict:
        if not shard_bytes:
            return {"imported": 0, "skipped": 0}
        shard = "/tmp/incoming_shard.db"
        Path(shard).write_bytes(shard_bytes)
        imported, skipped = self.db.merge_from(shard, recompute_hulls=True)
        Path(shard).unlink(missing_ok=True)
        self._persist()
        return {"imported": imported, "skipped": skipped}

    @modal.method()
    def snapshot(self, seed_rel: str) -> str:
        """Checkpoint the master DB and copy it to a stable seed file."""
        import shutil

        self.db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dest = Path(BURN_DIR) / seed_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(MASTER_DB, dest)
        volume.commit()
        return str(dest)

    @modal.method()
    def read_ledger(self) -> dict:
        import json

        path = Path(BURN_DIR) / "ledger.json"
        if path.exists():
            return json.loads(path.read_text())
        return {"gpu_seconds": 0.0, "est_cost": 0.0, "by_phase": {}}

    @modal.method()
    def bump_ledger(self, seconds: float, rate: float, phase: str) -> dict:
        import json

        path = Path(BURN_DIR) / "ledger.json"
        ledger = json.loads(path.read_text()) if path.exists() else {
            "gpu_seconds": 0.0,
            "est_cost": 0.0,
            "by_phase": {},
        }
        ledger["gpu_seconds"] += seconds
        ledger["est_cost"] += seconds / 3600.0 * rate
        ledger["by_phase"][phase] = ledger["by_phase"].get(phase, 0.0) + seconds
        path.write_text(json.dumps(ledger, indent=2))
        volume.commit()
        return ledger

    @modal.method()
    def phonon_candidates(self, cutoff: float, limit: int | None) -> list[dict]:
        query = """
            SELECT s.id, s.formula, s.num_atoms, s.cif_content,
                   MIN(h.e_above_hull) AS e_above_hull
            FROM structures s
            JOIN hull_entries h ON s.id = h.structure_id
            WHERE h.e_above_hull <= ?
              AND s.is_dynamically_stable IS NULL
              AND s.cif_content IS NOT NULL
            GROUP BY s.id
            ORDER BY e_above_hull ASC, s.energy_per_atom ASC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = self.db.conn.execute(query, (cutoff,)).fetchall()
        return [
            {
                "id": r["id"],
                "formula": r["formula"],
                "num_atoms": r["num_atoms"],
                "cif": r["cif_content"],
            }
            for r in rows
        ]

    @modal.method()
    def apply_phonons(self, results: list[dict]) -> int:
        applied = 0
        for res in results:
            if res.get("error") or "is_stable" not in res:
                continue
            self.db._update_structure(
                structure_id=res["id"],
                is_dynamically_stable=res["is_stable"],
                num_imaginary_modes=res["num_imaginary_modes"],
                min_phonon_frequency=res["min_frequency"],
                max_phonon_frequency=res["max_frequency"],
                phonon_supercell=tuple(res["supercell"]),
            )
            applied += 1
        self._persist()
        return applied

    @modal.method()
    def stats(self) -> dict:
        return self.db.get_statistics()

    @modal.method()
    def discoveries(self, near_hull_cutoff: float = 0.05) -> dict:
        from ggen.discoveries import summarize

        return summarize(self.db, near_hull_cutoff=near_hull_cutoff)


# ---------------------------------------------------------------------------
# Local driver.
# ---------------------------------------------------------------------------
def _normalize(system: str) -> str:
    return "-".join(sorted(system.replace("-", " ").split()))


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@app.local_entrypoint()
def main(
    stage: str = "all",
    palette: str = "",
    max_atoms: int = 16,
    min_atoms: int = 2,
    num_trials: int = 15,
    max_steps: int = 300,
    max_stoichiometries: int = 0,
    max_spend: float = 3700.0,
    phonon_cutoff: float = 0.05,
    phonon_limit: int = 0,
    phonon_batch_size: int = 8,
    round_size: int = 40,
    dry_run: bool = False,
):
    """Orchestrate the staged sweep + phonon backfill.

    stage: ``all`` | ``unary`` | ``binary`` | ``ternary`` | ``explore`` | ``phonon``
    GPU type and concurrency come from the GGEN_BURN_GPU / GGEN_BURN_MAX_CONTAINERS
    env vars (defaults: A10G, 15).
    """
    elements = (
        [e.strip() for e in palette.split(",") if e.strip()]
        if palette
        else DEFAULT_PALETTE
    )
    elements = sorted(set(elements))
    rate = GPU_HOURLY_RATE.get(BURN_GPU, GPU_HOURLY_RATE["A10G"])
    max_stoich = max_stoichiometries or None

    unaries = elements
    binaries = ["-".join(c) for c in itertools.combinations(elements, 2)]
    ternaries = ["-".join(c) for c in itertools.combinations(elements, 3)]

    print("=" * 64)
    print(f"GGEN BURN  ({len(elements)} elements, GPU={BURN_GPU} @ ${rate}/hr)")
    print("=" * 64)
    print(f"  Palette: {', '.join(elements)}")
    print(f"  Unaries: {len(unaries)}  Binaries: {len(binaries)}  Ternaries: {len(ternaries)}")
    print(
        f"  Stage: {stage}   Spend cap: ${max_spend:,.0f}   "
        f"Max containers: {BURN_MAX_CONTAINERS}"
    )
    est_explore_hr = len(ternaries) * 2.0 + len(binaries) * 0.5
    print(
        f"  Rough explore estimate: ~{est_explore_hr:,.0f} GPU-hr "
        f"(~${est_explore_hr * rate:,.0f}) before phonons"
    )

    if dry_run:
        print("\n[dry-run] Nothing launched.")
        return

    explore = explore_one
    phonons = phonons_batch
    master = MasterDB()

    ledger = master.read_ledger.remote()
    cost = ledger["est_cost"]
    done = set(_normalize(s) for s in master.done_systems.remote())
    print(f"  Resuming: {len(done)} systems already in DB, ${cost:,.2f} spent so far\n")

    def run_phase(name, systems, require_all, seed_rel):
        nonlocal cost
        todo = [s for s in systems if _normalize(s) not in done]
        if not todo:
            print(f"[{name}] nothing to do ({len(systems)} already present)")
            return True
        print(f"[{name}] {len(todo)}/{len(systems)} systems to explore")
        params = {
            "max_atoms": max_atoms,
            "min_atoms": min_atoms,
            "num_trials": num_trials,
            "max_steps": max_steps,
            "max_stoichiometries": max_stoich,
            "require_all_elements": require_all,
        }
        for rnd in _chunks(todo, round_size):
            if cost >= max_spend:
                print(f"[{name}] spend cap reached (${cost:,.2f}); stopping.")
                return False
            round_seconds = 0.0
            new_total = 0
            for res in explore.map(
                rnd,
                kwargs={"params": params, "seed_rel": seed_rel},
                order_outputs=False,
                return_exceptions=True,
            ):
                if isinstance(res, Exception):
                    print(f"    ! worker crashed: {res}")
                    continue
                round_seconds += res.get("elapsed_s", 0.0)
                if res.get("error"):
                    print(f"    ! {res['system']}: {res['error']}")
                    continue
                merged = master.merge_shard.remote(res["shard_bytes"])
                done.add(_normalize(res["system"]))
                new_total += merged["imported"]
            ledger = master.bump_ledger.remote(round_seconds, rate, name)
            cost = ledger["est_cost"]
            print(
                f"    round done: +{new_total} structures, "
                f"{round_seconds / 3600:.1f} GPU-hr, total ${cost:,.2f}"
            )
        return True

    want = {
        "all": {"unary", "binary", "ternary", "phonon"},
        "explore": {"unary", "binary", "ternary"},
        "unary": {"unary"},
        "binary": {"binary"},
        "ternary": {"ternary"},
        "phonon": {"phonon"},
    }[stage]

    if "unary" in want:
        if not run_phase("unary", unaries, require_all=False, seed_rel=None):
            return
        master.snapshot.remote("seed_binary.db")
    if "binary" in want:
        if not run_phase("binary", binaries, require_all=True, seed_rel="seed_binary.db"):
            return
        master.snapshot.remote("seed_ternary.db")
    if "ternary" in want:
        if not run_phase(
            "ternary", ternaries, require_all=True, seed_rel="seed_ternary.db"
        ):
            return

    if "phonon" in want:
        if cost >= max_spend:
            print(f"[phonon] spend cap reached (${cost:,.2f}); skipping.")
        else:
            run_phonons(
                master, phonons, rate, phonon_cutoff,
                phonon_limit or None, phonon_batch_size, max_spend,
            )

    stats = master.stats.remote()
    print("\n" + "=" * 64)
    print("BURN COMPLETE")
    print(f"  Structures: {stats.get('total_structures')}")
    print(f"  Unique formulas: {stats.get('unique_formulas')}")
    print(f"  Chemical systems: {stats.get('unique_chemsys')}")
    print(f"  Estimated spend: ${cost:,.2f}")

    disc = master.discoveries.remote(phonon_cutoff)
    print("\n" + "-" * 64)
    print(f"DISCOVERIES vs Materials Project (within {phonon_cutoff * 1000:.0f} meV)")
    print(f"  {disc['unique_discoveries']} unique non-elemental phases on hull "
          f"across {disc['systems_analyzed']} systems")
    print(f"    new compositions: {disc['new_compositions']}   beat MP: {disc['beat_mp']}")
    print(f"    phonon: {disc['phonon_stable']} stable / "
          f"{disc['phonon_unstable']} unstable / {disc['phonon_untested']} untested")
    print(f"    symmetry: {disc['higher_symmetry']} higher-symmetry / {disc['p1']} P1")
    if disc["top"]:
        print("\n  Most credible discoveries:")
        for d in disc["top"]:
            dyn = (
                "phonon OK" if d["is_dynamically_stable"] is True
                else "unstable" if d["is_dynamically_stable"] is False
                else "untested"
            )
            tag = "NEW" if d["category"] == "new_composition" else "BEAT-MP"
            print(
                f"    {d['formula']:<14} {(d['space_group'] or '?'):<10} "
                f"E={d['energy_per_atom']:.4f}  [{tag}] ({dyn})  {d['system']}"
            )
    print("\nFull report:  python scripts/discoveries.py --near-hull 0.05 -o discoveries.json")


def run_phonons(master, phonons, rate, cutoff, limit, batch_size, max_spend):
    candidates = master.phonon_candidates.remote(cutoff, limit)
    if not candidates:
        print("[phonon] no untested near-hull structures found.")
        return
    print(f"[phonon] {len(candidates)} structures within {cutoff * 1000:.0f} meV to validate")
    batches = list(_chunks(candidates, batch_size))
    stable = unstable = 0
    for results in phonons.map(batches, order_outputs=False, return_exceptions=True):
        if isinstance(results, Exception):
            print(f"    ! phonon worker crashed: {results}")
            continue
        seconds = sum(r.get("elapsed_s", 0.0) for r in results)
        applied = master.apply_phonons.remote(results)
        stable += sum(1 for r in results if r.get("is_stable") is True)
        unstable += sum(1 for r in results if r.get("is_stable") is False)
        ledger = master.bump_ledger.remote(seconds, rate, "phonon")
        print(
            f"    +{applied} validated (stable={stable}, unstable={unstable}), "
            f"total ${ledger['est_cost']:,.2f}"
        )
        if ledger["est_cost"] >= max_spend:
            print(f"[phonon] spend cap reached (${ledger['est_cost']:,.2f}); stopping.")
            return
