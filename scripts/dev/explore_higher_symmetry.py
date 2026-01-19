#!/usr/bin/env python3
"""
Follow-up to analyze_p1_symmetry.py:

If P1 structures have hidden higher symmetry, generate FRESH structures
in those space groups and see if we can find even lower energy configurations.

This takes a different approach than just idealizing:
1. Take a P1 structure with detected hidden symmetry
2. Generate new random structures in that space group (same composition)
3. Relax and compare energies

Run with: python scripts/dev/explore_higher_symmetry.py --input p1_symmetry_analysis.json
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from tqdm import tqdm

from ggen import GGen
from ggen.calculator import get_orb_calculator
from ggen.database import StructureDatabase

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class HigherSymmetryExplorationResult:
    """Result of generating new structures in a detected higher symmetry space group."""

    # Original P1 info
    original_structure_id: str
    formula: str
    original_energy_per_atom: float

    # Target space group (from symmetry detection)
    target_sg_number: int
    target_sg_symbol: str
    detection_tolerance: float

    # New generation results
    num_trials: int = 0
    num_successful: int = 0
    best_new_energy_per_atom: Optional[float] = None
    best_new_sg_number: Optional[int] = None
    best_new_sg_symbol: Optional[str] = None

    # All generated energies (for analysis)
    all_energies: List[float] = field(default_factory=list)
    all_space_groups: List[int] = field(default_factory=list)

    # Improvement
    energy_improvement: Optional[float] = None  # original - best_new (positive = improvement)
    found_lower_energy: bool = False

    error_message: Optional[str] = None


def generate_in_space_group(
    formula: str,
    target_sg: int,
    calculator,
    num_trials: int = 10,
    max_relax_steps: int = 200,
    fmax: float = 0.02,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Generate multiple structures in a specific space group.

    Returns list of dicts with energy_per_atom, structure, sg_number, sg_symbol.
    """
    results = []

    # Parse formula
    comp = Composition(formula)
    elements = [str(el) for el in comp.elements]
    counts = [int(comp[el]) for el in comp.elements]

    # Check if space group is compatible
    from pyxtal.symmetry import Group

    group = Group(target_sg, dim=3)
    is_compatible, _ = group.check_compatible(counts)

    if not is_compatible:
        logger.warning(f"Space group {target_sg} incompatible with {formula}")
        return results

    # Create GGen instance
    ggen = GGen(
        calculator=calculator,
        random_seed=random_seed,
        enable_trajectory=False,
    )

    for trial in range(num_trials):
        try:
            c = pyxtal()
            c.from_random(dim=3, group=target_sg, species=elements, numIons=counts)

            if not c.valid:
                continue

            # Relax
            from ase.filters import FrechetCellFilter
            from ase.optimize import LBFGS
            from pymatgen.io.ase import AseAtomsAdaptor

            atoms = c.to_ase()
            atoms.calc = calculator

            # Quick sanity check
            initial_energy = atoms.get_potential_energy()
            if initial_energy > 0:  # Likely overlapping atoms
                continue

            filtered = FrechetCellFilter(atoms)
            opt = LBFGS(filtered, logfile=None, maxstep=0.2)
            opt.run(fmax=fmax, steps=max_relax_steps)

            final_energy = atoms.get_potential_energy()
            energy_per_atom = final_energy / len(atoms)

            # Get final symmetry
            adaptor = AseAtomsAdaptor()
            relaxed_structure = adaptor.get_structure(atoms)

            try:
                analyzer = SpacegroupAnalyzer(relaxed_structure, symprec=0.1)
                final_sg = analyzer.get_space_group_number()
                final_sym = analyzer.get_space_group_symbol()
            except Exception:
                final_sg = 1
                final_sym = "P1"

            results.append({
                "energy_per_atom": energy_per_atom,
                "sg_number": final_sg,
                "sg_symbol": final_sym,
                "structure": relaxed_structure,
            })

        except Exception as e:
            logger.debug(f"Trial {trial} failed: {e}")
            continue

    return results


def explore_from_analysis(
    analysis_path: Path,
    db_path: Path,
    calculator,
    num_trials_per_structure: int = 10,
    max_relax_steps: int = 200,
    fmax: float = 0.02,
    min_improvement_threshold: float = 0.001,  # eV/atom
    save_to_db: bool = False,
) -> List[HigherSymmetryExplorationResult]:
    """
    Load P1 symmetry analysis results and explore higher-symmetry space groups.
    """
    # Load analysis
    with open(analysis_path) as f:
        analysis = json.load(f)

    results_data = analysis.get("results", [])

    # Filter to structures where higher symmetry was detected
    promising = [
        r for r in results_data
        if r.get("best_detected_sg_number", 1) > 1
    ]

    logger.info(f"Found {len(promising)} P1 structures with detected higher symmetry")

    if not promising:
        return []

    db = StructureDatabase(db_path) if save_to_db else None

    exploration_results = []

    for entry in tqdm(promising, desc="Exploring higher symmetry"):
        result = HigherSymmetryExplorationResult(
            original_structure_id=entry["structure_id"],
            formula=entry["formula"],
            original_energy_per_atom=entry["original_energy_per_atom"],
            target_sg_number=entry["best_detected_sg_number"],
            target_sg_symbol=entry["best_detected_sg_symbol"],
            detection_tolerance=entry["detection_tolerance"],
        )

        try:
            # Generate new structures in the detected space group
            gen_results = generate_in_space_group(
                formula=entry["formula"],
                target_sg=entry["best_detected_sg_number"],
                calculator=calculator,
                num_trials=num_trials_per_structure,
                max_relax_steps=max_relax_steps,
                fmax=fmax,
            )

            result.num_trials = num_trials_per_structure
            result.num_successful = len(gen_results)

            if gen_results:
                energies = [r["energy_per_atom"] for r in gen_results]
                result.all_energies = energies
                result.all_space_groups = [r["sg_number"] for r in gen_results]

                best_idx = np.argmin(energies)
                best = gen_results[best_idx]

                result.best_new_energy_per_atom = best["energy_per_atom"]
                result.best_new_sg_number = best["sg_number"]
                result.best_new_sg_symbol = best["sg_symbol"]

                improvement = result.original_energy_per_atom - best["energy_per_atom"]
                result.energy_improvement = improvement
                result.found_lower_energy = improvement > min_improvement_threshold

                # Save to database if improvement found
                if save_to_db and db and result.found_lower_energy:
                    structure = best["structure"]
                    stoichiometry = dict(Composition(entry["formula"]).get_el_amt_dict())
                    stoichiometry = {str(k): int(v) for k, v in stoichiometry.items()}

                    db.add_structure(
                        formula=entry["formula"],
                        stoichiometry=stoichiometry,
                        energy_per_atom=best["energy_per_atom"],
                        total_energy=best["energy_per_atom"] * len(structure),
                        num_atoms=len(structure),
                        space_group_number=best["sg_number"],
                        space_group_symbol=best["sg_symbol"],
                        structure=structure,
                        is_valid=True,
                        generation_metadata={
                            "source": "higher_symmetry_exploration",
                            "original_p1_id": entry["structure_id"],
                            "target_sg": entry["best_detected_sg_number"],
                            "detection_tolerance": entry["detection_tolerance"],
                        },
                    )
                    logger.info(
                        f"Saved improved structure for {entry['formula']}: "
                        f"{result.original_energy_per_atom:.4f} → {best['energy_per_atom']:.4f} eV/atom "
                        f"({improvement*1000:.1f} meV improvement)"
                    )

        except Exception as e:
            result.error_message = str(e)

        exploration_results.append(result)

    if db:
        db.close()

    return exploration_results


def print_exploration_summary(results: List[HigherSymmetryExplorationResult]):
    """Print summary of exploration results."""
    print("\n" + "=" * 80)
    print("HIGHER SYMMETRY EXPLORATION - SUMMARY")
    print("=" * 80)

    total = len(results)
    errors = sum(1 for r in results if r.error_message)
    valid = [r for r in results if not r.error_message and r.num_successful > 0]

    print(f"\nTotal structures explored: {total}")
    print(f"  Errors: {errors}")
    print(f"  Successful generations: {len(valid)}")

    if not valid:
        print("\nNo successful generations.")
        return

    # Success rate
    total_trials = sum(r.num_trials for r in valid)
    total_successful = sum(r.num_successful for r in valid)
    print(f"\nGeneration success rate: {total_successful}/{total_trials} ({100*total_successful/total_trials:.1f}%)")

    # Improvement statistics
    improved = [r for r in valid if r.found_lower_energy]
    print(f"\nFound lower energy: {len(improved)} / {len(valid)} ({100*len(improved)/len(valid):.1f}%)")

    if improved:
        improvements = [r.energy_improvement for r in improved if r.energy_improvement]
        avg_improvement = np.mean(improvements) * 1000  # meV
        max_improvement = max(improvements) * 1000

        print(f"Average improvement: {avg_improvement:.1f} meV/atom")
        print(f"Maximum improvement: {max_improvement:.1f} meV/atom")

        # Top improvements
        print("\n" + "-" * 40)
        print("TOP 10 IMPROVEMENTS")
        print("-" * 40)

        improved.sort(key=lambda r: -(r.energy_improvement or 0))
        print(f"\n{'Formula':<15} {'Orig E':<12} {'New E':<12} {'Δ (meV)':<10} {'Target SG':<15} {'Final SG':<15}")
        print("-" * 80)
        for r in improved[:10]:
            delta = (r.energy_improvement or 0) * 1000
            target_sg = f"{r.target_sg_symbol} (#{r.target_sg_number})"
            final_sg = f"{r.best_new_sg_symbol} (#{r.best_new_sg_number})"
            print(
                f"{r.formula:<15} {r.original_energy_per_atom:<12.4f} "
                f"{r.best_new_energy_per_atom:<12.4f} {delta:<10.1f} {target_sg:<15} {final_sg:<15}"
            )

    # Space group preservation
    preserved = sum(
        1 for r in valid
        if r.best_new_sg_number and r.best_new_sg_number >= r.target_sg_number
    )
    print(f"\nTarget symmetry preserved or improved: {preserved} / {len(valid)} ({100*preserved/len(valid):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate structures in detected higher-symmetry space groups"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to p1_symmetry_analysis JSON output",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="ggen.db",
        help="Path to the structure database",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per space group",
    )
    parser.add_argument(
        "--max-relax-steps",
        type=int,
        default=200,
        help="Maximum relaxation steps",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.02,
        help="Force convergence criterion (eV/Å)",
    )
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Save improved structures to the database",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Analysis file not found: {args.input}")

    db_path = Path(args.db)
    if not db_path.exists():
        ggen_dir = Path(__file__).parent.parent.parent
        db_path = ggen_dir / args.db

    logger.info(f"Loading analysis from: {input_path}")
    logger.info(f"Database: {db_path}")

    # Initialize calculator
    logger.info("Initializing ORB calculator...")
    calculator = get_orb_calculator()

    # Run exploration
    results = explore_from_analysis(
        analysis_path=input_path,
        db_path=db_path,
        calculator=calculator,
        num_trials_per_structure=args.trials,
        max_relax_steps=args.max_relax_steps,
        fmax=args.fmax,
        save_to_db=args.save_to_db,
    )

    # Print summary
    print_exploration_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"higher_symmetry_exploration_{ts}.json")

    output_data = {
        "parameters": {
            "input_analysis": str(input_path),
            "database": str(db_path),
            "trials_per_structure": args.trials,
            "max_relax_steps": args.max_relax_steps,
            "fmax": args.fmax,
            "save_to_db": args.save_to_db,
        },
        "summary": {
            "total_explored": len(results),
            "found_lower_energy": sum(1 for r in results if r.found_lower_energy),
            "avg_improvement_mev": np.mean([
                r.energy_improvement * 1000 for r in results
                if r.energy_improvement and r.found_lower_energy
            ]) if any(r.found_lower_energy for r in results) else 0,
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
