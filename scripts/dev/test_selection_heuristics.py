#!/usr/bin/env python3
"""
Empirical test: Do initial properties predict final relaxed energy?

This script generates many candidates, fully relaxes ALL of them,
and analyzes correlations between:
- Initial energy vs final energy
- Initial force magnitude vs final energy
- Initial stress vs final energy
- Mini-relaxation energy drop vs final energy

Run with: python scripts/test_selection_heuristics.py
"""


# TODO: see if https://smact.readthedocs.io/en/latest/examples/valence_electron_count.html can be used to score candidates

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from pyxtal import pyxtal
from tqdm import tqdm

from ggen import GGen
from ggen.calculator import get_orb_calculator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CandidateData:
    """Data collected for each candidate."""

    # Identification
    trial_idx: int
    space_group: int

    # Initial properties (pre-relaxation)
    initial_energy: float
    initial_force_rms: float
    initial_force_max: float
    initial_stress_norm: float
    initial_volume_per_atom: float

    # Mini-relaxation (5 steps)
    mini_relax_energy: Optional[float] = None
    mini_relax_delta: Optional[float] = None  # initial - mini (positive = dropped)

    # Full relaxation
    final_energy: Optional[float] = None
    final_steps: Optional[int] = None
    relaxation_failed: bool = False


def collect_initial_properties(atoms) -> dict:
    """Get energy, forces, stress from atoms object."""
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()  # Voigt notation, 6 components

    force_magnitudes = np.linalg.norm(forces, axis=1)

    return {
        "energy": energy,
        "force_rms": float(np.sqrt(np.mean(force_magnitudes**2))),
        "force_max": float(np.max(force_magnitudes)),
        "stress_norm": float(np.linalg.norm(stress)),
        "volume_per_atom": atoms.get_volume() / len(atoms),
    }


def mini_relax(atoms, steps=5) -> float:
    """Quick partial relaxation, return energy after."""
    atoms_copy = atoms.copy()
    atoms_copy.calc = atoms.calc

    filtered = FrechetCellFilter(atoms_copy)
    opt = LBFGS(filtered, logfile=None, maxstep=0.2)

    try:
        opt.run(fmax=0.5, steps=steps)  # Very loose, just get direction
        return atoms_copy.get_potential_energy()
    except Exception:
        return atoms_copy.get_potential_energy()


def full_relax(atoms, max_steps=200, fmax=0.02) -> tuple:
    """Full relaxation, return (final_energy, steps)."""
    atoms_copy = atoms.copy()
    atoms_copy.calc = atoms.calc

    filtered = FrechetCellFilter(atoms_copy)
    opt = LBFGS(filtered, logfile=None, maxstep=0.2)

    try:
        opt.run(fmax=fmax, steps=max_steps)
        return atoms_copy.get_potential_energy(), opt.nsteps
    except Exception as e:
        logger.warning(f"Relaxation failed: {e}")
        return None, None


def generate_candidates(
    formula: str,
    num_candidates: int,
    calculator,
    random_seed: Optional[int] = None,
) -> List[CandidateData]:
    """Generate candidates and collect all data."""

    ggen = GGen(calculator=calculator, random_seed=random_seed, enable_trajectory=False)

    # Parse formula
    from pymatgen.core import Composition

    comp = Composition(formula)
    elements = [str(el) for el in comp.elements]
    counts = [int(comp[el]) for el in comp.elements]

    # Get compatible space groups
    compatible = ggen.get_compatible_space_groups(elements, counts)
    if not compatible:
        raise ValueError(f"No compatible space groups for {formula}")

    # Select diverse space groups
    sg_numbers = [sg["number"] for sg in compatible]

    candidates = []
    trial_idx = 0
    attempts = 0
    max_attempts = num_candidates * 10  # Allow some failures

    pbar = tqdm(total=num_candidates, desc="Generating candidates")

    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1

        # Pick a random space group
        sg_number = np.random.choice(sg_numbers)

        # Generate random crystal
        c = pyxtal()
        try:
            c.from_random(dim=3, group=sg_number, species=elements, numIons=counts)
        except Exception:
            continue

        if not c.valid:
            continue

        # Convert to ASE and attach calculator
        try:
            atoms = c.to_ase()
            atoms.calc = calculator

            # Collect initial properties
            props = collect_initial_properties(atoms)

            # Skip if energy is crazy (likely overlapping atoms)
            if props["energy"] > 0 or props["force_max"] > 100:
                continue

            # Mini relaxation
            mini_energy = mini_relax(atoms, steps=5)
            mini_delta = props["energy"] - mini_energy

            # Full relaxation
            final_energy, final_steps = full_relax(atoms, max_steps=200, fmax=0.02)

            candidate = CandidateData(
                trial_idx=trial_idx,
                space_group=sg_number,
                initial_energy=props["energy"],
                initial_force_rms=props["force_rms"],
                initial_force_max=props["force_max"],
                initial_stress_norm=props["stress_norm"],
                initial_volume_per_atom=props["volume_per_atom"],
                mini_relax_energy=mini_energy,
                mini_relax_delta=mini_delta,
                final_energy=final_energy,
                final_steps=final_steps,
                relaxation_failed=(final_energy is None),
            )

            candidates.append(candidate)
            trial_idx += 1
            pbar.update(1)

        except Exception as e:
            logger.debug(f"Candidate failed: {e}")
            continue

    pbar.close()
    return candidates


def analyze_correlations(candidates: List[CandidateData]) -> dict:
    """Compute correlations between initial properties and final energy."""

    # Filter to successful relaxations
    valid = [c for c in candidates if not c.relaxation_failed]

    if len(valid) < 5:
        return {"error": "Not enough valid candidates for analysis"}

    # Extract arrays
    initial_e = np.array([c.initial_energy for c in valid])
    force_rms = np.array([c.initial_force_rms for c in valid])
    force_max = np.array([c.initial_force_max for c in valid])
    stress_norm = np.array([c.initial_stress_norm for c in valid])
    mini_delta = np.array([c.mini_relax_delta for c in valid])
    final_e = np.array([c.final_energy for c in valid])

    # Compute correlations with final energy
    def corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    correlations = {
        "initial_energy_vs_final": corr(initial_e, final_e),
        "force_rms_vs_final": corr(force_rms, final_e),
        "force_max_vs_final": corr(force_max, final_e),
        "stress_norm_vs_final": corr(stress_norm, final_e),
        "mini_delta_vs_final": corr(mini_delta, final_e),
    }

    # Statistics
    stats = {
        "num_candidates": len(valid),
        "num_failed": len(candidates) - len(valid),
        "final_energy_range": (float(np.min(final_e)), float(np.max(final_e))),
        "final_energy_std": float(np.std(final_e)),
    }

    # Ranking analysis: if we picked by initial_energy, how often would we get the best?
    best_final_idx = np.argmin(final_e)
    best_initial_idx = np.argmin(initial_e)
    best_mini_delta_idx = np.argmax(mini_delta)  # Biggest drop = potentially best

    ranking = {
        "best_by_final_energy": int(best_final_idx),
        "best_by_initial_energy": int(best_initial_idx),
        "best_by_mini_delta": int(best_mini_delta_idx),
        "initial_energy_picked_best": best_initial_idx == best_final_idx,
        "mini_delta_picked_best": best_mini_delta_idx == best_final_idx,
        # Rank of the true best when sorting by initial energy
        "true_best_rank_by_initial": int(
            np.where(np.argsort(initial_e) == best_final_idx)[0][0]
        )
        + 1,
        "true_best_rank_by_mini_delta": int(
            np.where(np.argsort(-mini_delta) == best_final_idx)[0][0]
        )
        + 1,
    }

    return {
        "correlations": correlations,
        "stats": stats,
        "ranking": ranking,
    }


def print_results(analysis: dict, candidates: List[CandidateData]):
    """Pretty print the analysis results."""

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS: Initial Properties vs Final Energy")
    print("=" * 60)

    corr = analysis["correlations"]
    print("\nCorrelations with final energy (closer to 1.0 = better predictor):")
    print(f"  Initial energy:     r = {corr['initial_energy_vs_final']:+.3f}")
    print(f"  Force RMS:          r = {corr['force_rms_vs_final']:+.3f}")
    print(f"  Force max:          r = {corr['force_max_vs_final']:+.3f}")
    print(f"  Stress norm:        r = {corr['stress_norm_vs_final']:+.3f}")
    print(f"  Mini-relax delta:   r = {corr['mini_delta_vs_final']:+.3f}")

    stats = analysis["stats"]
    print(f"\nStatistics:")
    print(f"  Valid candidates:   {stats['num_candidates']}")
    print(f"  Failed relaxations: {stats['num_failed']}")
    print(
        f"  Final E range:      {stats['final_energy_range'][0]:.2f} to {stats['final_energy_range'][1]:.2f} eV"
    )
    print(f"  Final E std:        {stats['final_energy_std']:.3f} eV")

    rank = analysis["ranking"]
    print(f"\nRanking analysis:")
    print(f"  Would initial_energy pick the best? {rank['initial_energy_picked_best']}")
    print(f"  Would mini_delta pick the best?     {rank['mini_delta_picked_best']}")
    print(
        f"  True best's rank by initial_energy: {rank['true_best_rank_by_initial']} / {stats['num_candidates']}"
    )
    print(
        f"  True best's rank by mini_delta:     {rank['true_best_rank_by_mini_delta']} / {stats['num_candidates']}"
    )

    # Show top 5 by final energy
    valid = [c for c in candidates if not c.relaxation_failed]
    sorted_by_final = sorted(valid, key=lambda x: x.final_energy)

    print(f"\nTop 5 by FINAL energy (ground truth):")
    print(
        f"  {'Rank':<5} {'SG':<5} {'Init E':<12} {'Force RMS':<12} {'Mini Δ':<12} {'Final E':<12}"
    )
    for i, c in enumerate(sorted_by_final[:5]):
        print(
            f"  {i+1:<5} {c.space_group:<5} {c.initial_energy:<12.3f} {c.initial_force_rms:<12.3f} {c.mini_relax_delta:<12.3f} {c.final_energy:<12.3f}"
        )

    print(f"\nTop 5 by INITIAL energy (current heuristic):")
    sorted_by_initial = sorted(valid, key=lambda x: x.initial_energy)
    for i, c in enumerate(sorted_by_initial[:5]):
        final_rank = sorted_by_final.index(c) + 1
        print(
            f"  {i+1:<5} {c.space_group:<5} {c.initial_energy:<12.3f} {c.initial_force_rms:<12.3f} {c.mini_relax_delta:<12.3f} {c.final_energy:<12.3f} (final rank: {final_rank})"
        )


def main():
    parser = argparse.ArgumentParser(description="Test selection heuristics")
    parser.add_argument("--formula", default="Fe2O3", help="Chemical formula to test")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=50,
        help="Number of candidates to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    logger.info(f"Testing selection heuristics for {args.formula}")
    logger.info(f"Generating {args.num_candidates} candidates...")

    # Initialize calculator
    calculator = get_orb_calculator()

    # Set random seed
    np.random.seed(args.seed)

    # Generate and analyze
    candidates = generate_candidates(
        formula=args.formula,
        num_candidates=args.num_candidates,
        calculator=calculator,
        random_seed=args.seed,
    )

    analysis = analyze_correlations(candidates)
    print_results(analysis, candidates)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"heuristic_test_{args.formula}_{ts}.json")

    results = {
        "formula": args.formula,
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "analysis": analysis,
        "candidates": [asdict(c) for c in candidates],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
