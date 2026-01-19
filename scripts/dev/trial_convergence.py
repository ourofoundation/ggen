#!/usr/bin/env python3
"""
Test how many trials are needed to reliably find the ground state.

For a given formula, generate N trials across space groups and track:
- How does best-found energy improve with more trials?
- At what point do we stop finding improvements?
- What's the variance in outcomes with different random seeds?

This helps determine optimal num_trials for exploration.

Run with: python scripts/dev/trial_convergence.py --formula Co2Fe15 --max-trials 100
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from tqdm import tqdm

from ggen.calculator import get_orb_calculator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    """Result of a single trial."""
    trial_idx: int
    sg_number: int
    sg_symbol: str
    energy_per_atom: float
    relax_steps: int
    success: bool = True
    error: Optional[str] = None


@dataclass
class ConvergenceResult:
    """Result of convergence analysis."""
    formula: str
    total_trials: int
    successful_trials: int
    
    # All trial results
    trials: List[TrialResult] = field(default_factory=list)
    
    # Convergence curve: best energy found after N trials
    convergence_curve: List[Tuple[int, float]] = field(default_factory=list)
    
    # Final statistics
    final_best_energy: Optional[float] = None
    final_best_sg: Optional[int] = None
    final_best_sg_symbol: Optional[str] = None
    
    # At what trial did we find the final best?
    best_found_at_trial: Optional[int] = None
    
    # Energy statistics
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None
    energy_min: Optional[float] = None
    energy_max: Optional[float] = None


def get_compatible_space_groups(counts: List[int]) -> List[int]:
    """Get all compatible space group numbers."""
    compatible = []
    for sg_number in range(1, 231):
        try:
            group = Group(sg_number, dim=3)
            is_compatible, _ = group.check_compatible(counts)
            if is_compatible:
                compatible.append(sg_number)
        except Exception:
            continue
    return compatible


def run_single_trial(
    elements: List[str],
    counts: List[int],
    sg_number: int,
    calculator,
    max_relax_steps: int = 200,
    fmax: float = 0.02,
) -> Optional[Tuple[float, int]]:
    """
    Run a single structure generation + relaxation trial.
    
    Returns (energy_per_atom, relax_steps) or None if failed.
    """
    try:
        c = pyxtal()
        c.from_random(dim=3, group=sg_number, species=elements, numIons=counts)
        
        if not c.valid:
            return None
        
        atoms = c.to_ase()
        atoms.calc = calculator
        
        # Sanity check
        initial_energy = atoms.get_potential_energy()
        if initial_energy > 0:
            return None
        
        # Relax
        filtered = FrechetCellFilter(atoms)
        opt = LBFGS(filtered, logfile=None, maxstep=0.2)
        opt.run(fmax=fmax, steps=max_relax_steps)
        
        final_energy = atoms.get_potential_energy()
        energy_per_atom = final_energy / len(atoms)
        
        return energy_per_atom, opt.nsteps
        
    except Exception:
        return None


def run_convergence_test(
    formula: str,
    calculator,
    max_trials: int = 100,
    max_relax_steps: int = 200,
    fmax: float = 0.02,
    sampling_strategy: str = "uniform",  # "uniform", "high_symmetry_bias"
    random_seed: Optional[int] = None,
) -> ConvergenceResult:
    """
    Run convergence test for a formula.
    
    Args:
        formula: Chemical formula
        max_trials: Maximum number of trials to run
        sampling_strategy: How to sample space groups
            - "uniform": equal probability for all compatible SGs
            - "high_symmetry_bias": prefer higher space group numbers
    """
    result = ConvergenceResult(
        formula=formula,
        total_trials=max_trials,
        successful_trials=0,
    )
    
    # Parse formula
    comp = Composition(formula)
    elements = [str(el) for el in comp.elements]
    counts = [int(comp[el]) for el in comp.elements]
    
    # Get compatible space groups
    compatible_sgs = get_compatible_space_groups(counts)
    if not compatible_sgs:
        return result
    
    logger.info(f"Formula {formula}: {len(compatible_sgs)} compatible space groups")
    
    # Set up random generator
    rng = np.random.default_rng(random_seed)
    
    # Prepare space group sampling weights
    if sampling_strategy == "high_symmetry_bias":
        # Weight by space group number (higher = more weight)
        weights = np.array([sg / 230.0 for sg in compatible_sgs])
        weights = weights / weights.sum()
    else:
        weights = None  # Uniform
    
    # Run trials
    best_energy = float('inf')
    best_sg = None
    best_sg_symbol = None
    best_found_at = None
    
    energies = []
    
    for trial_idx in tqdm(range(max_trials), desc=f"Trials for {formula}"):
        # Sample a space group
        if weights is not None:
            sg_number = rng.choice(compatible_sgs, p=weights)
        else:
            sg_number = rng.choice(compatible_sgs)
        
        # Run trial
        trial_result = run_single_trial(
            elements=elements,
            counts=counts,
            sg_number=sg_number,
            calculator=calculator,
            max_relax_steps=max_relax_steps,
            fmax=fmax,
        )
        
        if trial_result is None:
            result.trials.append(TrialResult(
                trial_idx=trial_idx,
                sg_number=sg_number,
                sg_symbol=Group(sg_number, dim=3).symbol,
                energy_per_atom=0,
                relax_steps=0,
                success=False,
            ))
            continue
        
        energy, steps = trial_result
        sg_symbol = Group(sg_number, dim=3).symbol
        
        result.trials.append(TrialResult(
            trial_idx=trial_idx,
            sg_number=sg_number,
            sg_symbol=sg_symbol,
            energy_per_atom=energy,
            relax_steps=steps,
            success=True,
        ))
        
        result.successful_trials += 1
        energies.append(energy)
        
        # Track best
        if energy < best_energy:
            best_energy = energy
            best_sg = sg_number
            best_sg_symbol = sg_symbol
            best_found_at = trial_idx
        
        # Record convergence curve
        result.convergence_curve.append((trial_idx + 1, best_energy))
    
    # Final stats
    if energies:
        result.final_best_energy = best_energy
        result.final_best_sg = best_sg
        result.final_best_sg_symbol = best_sg_symbol
        result.best_found_at_trial = best_found_at
        
        result.energy_mean = float(np.mean(energies))
        result.energy_std = float(np.std(energies))
        result.energy_min = float(np.min(energies))
        result.energy_max = float(np.max(energies))
    
    return result


def print_convergence_analysis(result: ConvergenceResult):
    """Print analysis of convergence results."""
    print("\n" + "=" * 70)
    print(f"CONVERGENCE ANALYSIS: {result.formula}")
    print("=" * 70)
    
    print(f"\nTrials: {result.successful_trials} / {result.total_trials} successful")
    
    if not result.convergence_curve:
        print("No successful trials!")
        return
    
    print(f"\nFinal best energy: {result.final_best_energy:.4f} eV/atom")
    print(f"Best space group: {result.final_best_sg_symbol} (#{result.final_best_sg})")
    print(f"Found at trial: {result.best_found_at_trial + 1}")
    
    print(f"\nEnergy statistics across all trials:")
    print(f"  Mean: {result.energy_mean:.4f} eV/atom")
    print(f"  Std:  {result.energy_std:.4f} eV/atom")
    print(f"  Min:  {result.energy_min:.4f} eV/atom")
    print(f"  Max:  {result.energy_max:.4f} eV/atom")
    
    # Show convergence at key points
    print(f"\nConvergence curve (best energy found after N trials):")
    checkpoints = [5, 10, 15, 20, 30, 50, 75, 100]
    
    print(f"  {'Trials':<10} {'Best E':<15} {'Δ from final':<15}")
    print("  " + "-" * 40)
    
    for n in checkpoints:
        if n <= len(result.convergence_curve):
            _, best_at_n = result.convergence_curve[n - 1]
            delta = (best_at_n - result.final_best_energy) * 1000  # meV
            print(f"  {n:<10} {best_at_n:<15.4f} {delta:<15.1f} meV")
    
    # What percentage of improvement came after N trials?
    print(f"\n% of best energy found by N trials:")
    initial_energy = result.convergence_curve[0][1]
    total_improvement = initial_energy - result.final_best_energy
    
    if total_improvement > 0.001:
        for n in [5, 10, 15, 20, 30, 50]:
            if n <= len(result.convergence_curve):
                _, best_at_n = result.convergence_curve[n - 1]
                improvement_at_n = initial_energy - best_at_n
                pct = 100 * improvement_at_n / total_improvement
                print(f"  After {n} trials: {pct:.1f}%")
    
    # Space group distribution of successful trials
    print(f"\nSpace group distribution of successful trials:")
    sg_counts: Dict[str, Tuple[int, List[float]]] = {}
    for t in result.trials:
        if t.success:
            key = f"{t.sg_symbol} (#{t.sg_number})"
            if key not in sg_counts:
                sg_counts[key] = (0, [])
            count, energies = sg_counts[key]
            sg_counts[key] = (count + 1, energies + [t.energy_per_atom])
    
    # Sort by best energy found in that SG
    sorted_sgs = sorted(
        sg_counts.items(),
        key=lambda x: min(x[1][1])
    )
    
    print(f"  {'Space Group':<20} {'Count':<8} {'Best E':<12} {'Mean E':<12}")
    print("  " + "-" * 55)
    for sg, (count, energies) in sorted_sgs[:10]:
        best_e = min(energies)
        mean_e = np.mean(energies)
        print(f"  {sg:<20} {count:<8} {best_e:<12.4f} {mean_e:<12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test convergence of structure search with number of trials"
    )
    parser.add_argument(
        "--formula",
        type=str,
        required=True,
        help="Chemical formula to test (e.g., 'Co2Fe15')",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=100,
        help="Maximum number of trials to run",
    )
    parser.add_argument(
        "--max-relax-steps",
        type=int,
        default=200,
        help="Maximum relaxation steps per trial",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.02,
        help="Force convergence criterion (eV/Å)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["uniform", "high_symmetry_bias"],
        default="uniform",
        help="Space group sampling strategy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    args = parser.parse_args()
    
    # Initialize calculator
    logger.info("Initializing ORB calculator...")
    calculator = get_orb_calculator()
    
    # Run convergence test
    logger.info(f"\nRunning convergence test for {args.formula}")
    logger.info(f"Max trials: {args.max_trials}")
    logger.info(f"Strategy: {args.strategy}")
    
    result = run_convergence_test(
        formula=args.formula,
        calculator=calculator,
        max_trials=args.max_trials,
        max_relax_steps=args.max_relax_steps,
        fmax=args.fmax,
        sampling_strategy=args.strategy,
        random_seed=args.seed,
    )
    
    # Print analysis
    print_convergence_analysis(result)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        formula_clean = args.formula.replace(" ", "_")
        output_path = Path(f"convergence_{formula_clean}_{ts}.json")
    
    output_data = {
        "parameters": {
            "formula": args.formula,
            "max_trials": args.max_trials,
            "max_relax_steps": args.max_relax_steps,
            "fmax": args.fmax,
            "strategy": args.strategy,
            "seed": args.seed,
        },
        "summary": {
            "total_trials": result.total_trials,
            "successful_trials": result.successful_trials,
            "final_best_energy": result.final_best_energy,
            "final_best_sg": int(result.final_best_sg) if result.final_best_sg else None,
            "best_found_at_trial": int(result.best_found_at_trial) if result.best_found_at_trial else None,
            "energy_mean": result.energy_mean,
            "energy_std": result.energy_std,
        },
        "convergence_curve": [(int(n), float(e)) for n, e in result.convergence_curve],
        "trials": [
            {
                "trial_idx": int(t.trial_idx),
                "sg_number": int(t.sg_number),
                "sg_symbol": t.sg_symbol,
                "energy_per_atom": float(t.energy_per_atom) if t.energy_per_atom else 0,
                "relax_steps": int(t.relax_steps),
                "success": t.success,
            }
            for t in result.trials
        ],
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
