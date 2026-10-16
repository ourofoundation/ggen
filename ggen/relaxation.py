"""Batched ORB v3 relaxation with torch-sim, one fixed recipe.

ORB v3 conservative, FIRE with a Frechet cell filter, forces to 0.02 eV/Å,
at most 400 steps per structure. Structures of any composition and size share
one GPU run; when the pool holds more atoms than ``BATCH_ATOMS`` they stream
through an in-flight batch, each converged one swapped for a waiting one.

This is the recipe Matra-Genoa and GenBench relax with. GenBench scores every
model's output with it, so changing a constant here changes benchmark results:
bump ``RECIPE`` when you do.

Imports torch lazily, so the module can be imported anywhere ggen is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from pymatgen.core import Structure

RECIPE = "orb-v3-conservative-inf-mpa/fire-frechet/fmax0.02/steps400"

ORB_MODEL = "orb_v3_conservative_inf_mpa"
ORB_PRECISION = "float32-high"
ORB_EDGE_METHOD = "knn_alchemi"
# Eager: with torch.compile, 256 LiFePO4 candidates relaxed in 173 s instead
# of 139 s and 64 Fe2O3 in 95 s instead of 32 s (A10G, 2026-10-15), since
# graph sizes change every step and keep recompiling.
ORB_COMPILE = False
FMAX = 0.02
MAX_STEPS = 400
# Atoms on the GPU at once. 4,000 brushed the A10G's 24 GB with sulfides
# (ORB v3 with stress needs ~5-6 MiB/atom), so this leaves headroom.
BATCH_ATOMS = 3000
STEPS_BETWEEN_SWAPS = 5
# Relaxed structures with atoms closer than this (Å) have collapsed.
MIN_DISTANCE = 0.5


def load_relaxer(device: str = "cuda"):
    """ORB v3 conservative wrapped for torch-sim."""
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

    orbff, adapter = getattr(pretrained, ORB_MODEL)(
        device=device, precision=ORB_PRECISION, compile=ORB_COMPILE
    )
    return OrbTorchSimModel(orbff, adapter, edge_method=ORB_EDGE_METHOD, device=device)


def _in_flight_batcher(model, max_atoms: float, max_iterations: int):
    import torch_sim as ts

    class _PinnedBatcher(ts.InFlightAutoBatcher):
        """In-flight autobatcher whose per-system step cap can't be overwritten.

        ``ts.optimize`` resets ``max_iterations`` from its global ``max_steps``
        and, past that, takes one step per swap, starving late systems in a
        pool larger than one batch. Pinning the cap gives every system its own
        budget (same fix as ``ggen.ggen._InFlightBatcher``).
        """

        @property
        def max_iterations(self):
            return self._pinned_max_iterations

        @max_iterations.setter
        def max_iterations(self, value):
            if not hasattr(self, "_pinned_max_iterations"):
                self._pinned_max_iterations = value

    return _PinnedBatcher(
        model=model,
        memory_scales_with="n_atoms",
        max_memory_scaler=max_atoms,
        max_iterations=max_iterations,
    )


@dataclass
class Relaxed:
    structure: Structure
    energy_per_atom: float
    fmax: float

    @property
    def converged(self) -> bool:
        return self.fmax <= FMAX * 1.05


def relax(ts_model, structures: Sequence[Structure]) -> List[Relaxed]:
    """Relax every structure in one torch-sim run, keeping input order."""
    import torch_sim as ts
    from pymatgen.io.ase import AseAtomsAdaptor

    if not structures:
        return []
    atoms_list = [AseAtomsAdaptor.get_atoms(s) for s in structures]
    # ORB's torch-sim adapter reads charge and spin from the state.
    for atoms in atoms_list:
        atoms.info.setdefault("charge", 0.0)
        atoms.info.setdefault("spin", 1.0)
    state = ts.io.atoms_to_state(
        atoms_list,
        ts_model.device,
        ts_model.dtype,
        system_extras_map={"charge": "charge", "spin": "spin"},
    )

    total_atoms = sum(len(a) for a in atoms_list)
    autobatcher = False
    max_steps = MAX_STEPS
    if total_atoms > BATCH_ATOMS:
        largest = max(len(a) for a in atoms_list)
        autobatcher = _in_flight_batcher(
            ts_model,
            float(max(BATCH_ATOMS, largest)),
            max(1, MAX_STEPS // STEPS_BETWEEN_SWAPS),
        )
        # The per-structure cap lives on the autobatcher.
        max_steps = 10**9

    final = ts.optimize(
        system=state,
        model=ts_model,
        optimizer=ts.Optimizer.fire,
        max_steps=max_steps,
        steps_between_swaps=STEPS_BETWEEN_SWAPS,
        autobatcher=autobatcher,
        init_kwargs={"cell_filter": ts.CellFilter.frechet},
        convergence_fn=ts.generate_force_convergence_fn(force_tol=FMAX),
    )
    energies = final.energy.detach().cpu().numpy()
    fmaxes = ts.system_wise_max_force(final).detach().cpu().numpy()
    out = []
    for i, atoms in enumerate(final.to_atoms()):
        s = AseAtomsAdaptor.get_structure(atoms)
        s = Structure(s.lattice, s.species, s.frac_coords, to_unit_cell=True)
        out.append(Relaxed(s, float(energies[i]) / len(s), float(fmaxes[i])))
    return out


def is_sane(r: Relaxed) -> bool:
    """Finite energy and forces, and no atoms collapsed onto each other."""
    if not math.isfinite(r.energy_per_atom) or not math.isfinite(r.fmax):
        return False
    if len(r.structure) < 2:
        return True
    dists = np.array(r.structure.distance_matrix)
    # Sites on top of each other are 0 apart, so mask the diagonal rather
    # than dropping zeros.
    np.fill_diagonal(dists, np.inf)
    return float(dists.min()) >= MIN_DISTANCE
