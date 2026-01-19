#!/usr/bin/env python3
"""
Challenge P1 ground states by searching higher-symmetry space groups.

Hypothesis: P1 has too many degrees of freedom, leading to local minima traps.
By explicitly searching higher-symmetry space groups, we might find lower-energy
configurations that were missed.

For each formula where P1 is currently the best structure:
1. Get all compatible higher-symmetry space groups
2. Generate multiple structures in each space group
3. Relax and compare to the P1 energy
4. Report if any beat the P1

Run with: python scripts/dev/challenge_p1_ground_states.py --chemsys Co-Fe-Ge
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from pymatgen.core import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from tqdm import tqdm

from ggen.calculator import get_orb_calculator
from ggen.database import StructureDatabase, StoredStructure

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Crystal system hierarchy (higher = more symmetric)
CRYSTAL_SYSTEM_RANK = {
    "triclinic": 0,
    "monoclinic": 1,
    "orthorhombic": 2,
    "tetragonal": 3,
    "trigonal": 3,
    "hexagonal": 4,
    "cubic": 5,
}


def get_crystal_system(sg_number: int) -> str:
    """Get crystal system from space group number."""
    if sg_number <= 2:
        return "triclinic"
    elif sg_number <= 15:
        return "monoclinic"
    elif sg_number <= 74:
        return "orthorhombic"
    elif sg_number <= 142:
        return "tetragonal"
    elif sg_number <= 167:
        return "trigonal"
    elif sg_number <= 194:
        return "hexagonal"
    else:
        return "cubic"


def get_compatible_space_groups(counts: List[int], min_sg: int = 3) -> List[Dict]:
    """
    Get space groups compatible with given atom counts, filtered to higher symmetry.

    Args:
        counts: List of atom counts per element
        min_sg: Minimum space group number (default 3 = skip triclinic)

    Returns:
        List of dicts with space group info, sorted by sg_number descending
    """
    compatible = []

    for sg_number in range(min_sg, 231):
        try:
            group = Group(sg_number, dim=3)
            is_compatible, _ = group.check_compatible(counts)
            if is_compatible:
                compatible.append(
                    {
                        "number": sg_number,
                        "symbol": group.symbol,
                        "crystal_system": get_crystal_system(sg_number),
                    }
                )
        except Exception:
            continue

    # Sort by space group number descending (higher symmetry first)
    compatible.sort(key=lambda x: x["number"], reverse=True)
    return compatible


def sample_space_groups(
    compatible: List[Dict],
    max_per_system: int = 3,
    prioritize_high_symmetry: bool = True,
) -> List[int]:
    """
    Sample space groups to try, ensuring diversity across crystal systems.

    Args:
        compatible: List of compatible space group dicts
        max_per_system: Max space groups to sample from each crystal system
        prioritize_high_symmetry: If True, take highest SG numbers from each system

    Returns:
        List of space group numbers to try
    """
    by_system = defaultdict(list)
    for sg in compatible:
        by_system[sg["crystal_system"]].append(sg["number"])

    selected = []

    # Process systems from highest to lowest symmetry
    systems_ordered = sorted(
        by_system.keys(), key=lambda s: CRYSTAL_SYSTEM_RANK.get(s, 0), reverse=True
    )

    for system in systems_ordered:
        sgs = by_system[system]
        if prioritize_high_symmetry:
            # Take highest space group numbers
            sgs = sorted(sgs, reverse=True)
        else:
            # Random sample
            np.random.shuffle(sgs)

        selected.extend(sgs[:max_per_system])

    return selected


@dataclass
class SpaceGroupTrialResult:
    """Result of trying a single space group."""

    sg_number: int
    sg_symbol: str
    crystal_system: str
    num_trials: int
    num_successful: int
    best_energy_per_atom: Optional[float] = None
    all_energies: List[float] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class P1ChallengeResult:
    """Result of challenging a P1 ground state."""

    formula: str
    p1_energy_per_atom: float
    p1_structure_id: str

    # Compatible space groups found
    num_compatible_sg: int = 0
    space_groups_tried: List[int] = field(default_factory=list)

    # Results per space group
    sg_results: List[SpaceGroupTrialResult] = field(default_factory=list)

    # Best challenger
    best_challenger_sg: Optional[int] = None
    best_challenger_symbol: Optional[str] = None
    best_challenger_energy: Optional[float] = None

    # Outcome
    p1_beaten: bool = False
    improvement: Optional[float] = (
        None  # P1_energy - best_challenger (positive = P1 beaten)
    )

    error_message: Optional[str] = None


def generate_in_space_group(
    elements: List[str],
    counts: List[int],
    sg_number: int,
    calculator,
    num_trials: int = 5,
    max_relax_steps: int = 200,
    fmax: float = 0.02,
) -> SpaceGroupTrialResult:
    """Generate and relax structures in a specific space group."""

    group = Group(sg_number, dim=3)
    result = SpaceGroupTrialResult(
        sg_number=sg_number,
        sg_symbol=group.symbol,
        crystal_system=get_crystal_system(sg_number),
        num_trials=num_trials,
        num_successful=0,
    )

    adaptor = AseAtomsAdaptor()
    energies = []

    for trial in range(num_trials):
        try:
            c = pyxtal()
            c.from_random(dim=3, group=sg_number, species=elements, numIons=counts)

            if not c.valid:
                continue

            atoms = c.to_ase()
            atoms.calc = calculator

            # Quick sanity check
            try:
                initial_energy = atoms.get_potential_energy()
                if initial_energy > 0:  # Likely overlapping atoms
                    continue
            except Exception:
                continue

            # Relax
            filtered = FrechetCellFilter(atoms)
            opt = LBFGS(filtered, logfile=None, maxstep=0.2)
            opt.run(fmax=fmax, steps=max_relax_steps)

            final_energy = atoms.get_potential_energy()
            energy_per_atom = final_energy / len(atoms)

            energies.append(energy_per_atom)
            result.num_successful += 1

        except Exception as e:
            continue

    if energies:
        result.all_energies = energies
        result.best_energy_per_atom = min(energies)

    return result


def challenge_p1_formula(
    formula: str,
    p1_energy: float,
    p1_id: str,
    calculator,
    max_sg_to_try: int = 15,
    trials_per_sg: int = 5,
    max_relax_steps: int = 200,
    fmax: float = 0.02,
    skip_monoclinic: bool = False,
) -> P1ChallengeResult:
    """Challenge a P1 ground state with higher-symmetry space groups."""

    result = P1ChallengeResult(
        formula=formula,
        p1_energy_per_atom=p1_energy,
        p1_structure_id=p1_id,
    )

    # Parse formula
    comp = Composition(formula)
    elements = [str(el) for el in comp.elements]
    counts = [int(comp[el]) for el in comp.elements]

    # Get compatible higher-symmetry space groups
    min_sg = 16 if skip_monoclinic else 3  # Skip monoclinic if desired
    compatible = get_compatible_space_groups(counts, min_sg=min_sg)
    result.num_compatible_sg = len(compatible)

    if not compatible:
        result.error_message = "No compatible higher-symmetry space groups"
        return result

    # Sample space groups to try
    sg_numbers = sample_space_groups(compatible, max_per_system=3)[:max_sg_to_try]
    result.space_groups_tried = sg_numbers

    # Try each space group
    best_energy = p1_energy
    best_sg = None
    best_symbol = None

    for sg_num in sg_numbers:
        sg_result = generate_in_space_group(
            elements=elements,
            counts=counts,
            sg_number=sg_num,
            calculator=calculator,
            num_trials=trials_per_sg,
            max_relax_steps=max_relax_steps,
            fmax=fmax,
        )
        result.sg_results.append(sg_result)

        if sg_result.best_energy_per_atom is not None:
            if sg_result.best_energy_per_atom < best_energy:
                best_energy = sg_result.best_energy_per_atom
                best_sg = sg_result.sg_number
                best_symbol = sg_result.sg_symbol

    # Check if P1 was beaten
    if best_sg is not None:
        result.best_challenger_sg = best_sg
        result.best_challenger_symbol = best_symbol
        result.best_challenger_energy = best_energy
        result.improvement = p1_energy - best_energy
        result.p1_beaten = result.improvement > 0.001  # 1 meV threshold

    return result


def get_p1_ground_states(
    db: StructureDatabase,
    chemsys: str,
) -> List[Tuple[str, float, str]]:
    """
    Get formulas where P1 is the lowest-energy structure.

    Returns:
        List of (formula, energy_per_atom, structure_id) tuples
    """
    # Get all structures for this chemical system
    all_structures = db.get_structures_for_subsystem(chemsys, valid_only=True)

    # Group by formula and find best for each
    by_formula: Dict[str, List[StoredStructure]] = defaultdict(list)
    for s in all_structures:
        by_formula[s.formula].append(s)

    # Find formulas where P1 is best
    p1_ground_states = []

    for formula, structures in by_formula.items():
        # Sort by energy
        structures.sort(key=lambda x: x.energy_per_atom)
        best = structures[0]

        # Check if best is P1
        if best.space_group_number == 1 or best.space_group_symbol == "P1":
            p1_ground_states.append((formula, best.energy_per_atom, best.id))

    logger.info(
        f"Found {len(p1_ground_states)} formulas with P1 ground state out of {len(by_formula)} total"
    )

    return p1_ground_states


def print_summary(results: List[P1ChallengeResult]):
    """Print summary of challenge results."""
    print("\n" + "=" * 80)
    print("P1 GROUND STATE CHALLENGE - SUMMARY")
    print("=" * 80)

    total = len(results)
    errors = sum(1 for r in results if r.error_message)
    valid = [r for r in results if not r.error_message]

    print(f"\nTotal P1 ground states challenged: {total}")
    print(f"  Errors: {errors}")
    print(f"  Valid challenges: {len(valid)}")

    if not valid:
        print("\nNo valid results.")
        return

    # How many were beaten?
    beaten = [r for r in valid if r.p1_beaten]
    print(
        f"\nP1 beaten by higher symmetry: {len(beaten)} / {len(valid)} ({100*len(beaten)/len(valid):.1f}%)"
    )

    if beaten:
        improvements = [r.improvement for r in beaten if r.improvement]
        avg_improvement = np.mean(improvements) * 1000
        max_improvement = max(improvements) * 1000

        print(f"Average improvement when beaten: {avg_improvement:.1f} meV/atom")
        print(f"Maximum improvement: {max_improvement:.1f} meV/atom")

        # Which space groups beat P1?
        sg_wins: Dict[str, int] = defaultdict(int)
        for r in beaten:
            key = f"{r.best_challenger_symbol} (#{r.best_challenger_sg})"
            sg_wins[key] += 1

        print("\nWinning space groups:")
        for sg, count in sorted(sg_wins.items(), key=lambda x: -x[1]):
            print(f"  {sg}: {count}")

        # Top improvements
        print("\n" + "-" * 60)
        print("TOP 10 IMPROVEMENTS (P1 beaten)")
        print("-" * 60)

        beaten.sort(key=lambda r: -(r.improvement or 0))
        print(
            f"\n{'Formula':<15} {'P1 E':<12} {'New E':<12} {'Δ (meV)':<10} {'Winner SG':<20}"
        )
        print("-" * 70)
        for r in beaten[:10]:
            delta = (r.improvement or 0) * 1000
            winner = f"{r.best_challenger_symbol} (#{r.best_challenger_sg})"
            print(
                f"{r.formula:<15} {r.p1_energy_per_atom:<12.4f} {r.best_challenger_energy:<12.4f} {delta:<10.1f} {winner:<20}"
            )

    # Statistics on compatible space groups
    avg_compatible = np.mean([r.num_compatible_sg for r in valid])
    avg_tried = np.mean([len(r.space_groups_tried) for r in valid])
    print(f"\nAvg compatible SGs per formula: {avg_compatible:.1f}")
    print(f"Avg SGs tried per formula: {avg_tried:.1f}")

    # Success rate by crystal system
    print("\n" + "-" * 60)
    print("SUCCESS RATE BY CRYSTAL SYSTEM")
    print("-" * 60)

    system_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "successful": 0, "beat_p1": 0}
    )

    for r in valid:
        for sg_result in r.sg_results:
            system = sg_result.crystal_system
            system_stats[system]["total"] += 1
            if sg_result.num_successful > 0:
                system_stats[system]["successful"] += 1
            if (
                sg_result.best_energy_per_atom
                and sg_result.best_energy_per_atom < r.p1_energy_per_atom - 0.001
            ):
                system_stats[system]["beat_p1"] += 1

    print(f"\n{'System':<15} {'Tried':<8} {'Generated':<12} {'Beat P1':<10}")
    print("-" * 50)
    for system in [
        "cubic",
        "hexagonal",
        "tetragonal",
        "trigonal",
        "orthorhombic",
        "monoclinic",
    ]:
        stats = system_stats[system]
        if stats["total"] > 0:
            print(
                f"{system:<15} {stats['total']:<8} {stats['successful']:<12} {stats['beat_p1']:<10}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Challenge P1 ground states with higher-symmetry space groups"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="ggen.db",
        help="Path to the structure database",
    )
    parser.add_argument(
        "--chemsys",
        type=str,
        required=True,
        help="Chemical system to analyze (e.g., 'Co-Fe-Ge')",
    )
    parser.add_argument(
        "--max-formulas",
        type=int,
        default=20,
        help="Maximum number of P1 formulas to challenge",
    )
    parser.add_argument(
        "--max-sg",
        type=int,
        default=15,
        help="Maximum space groups to try per formula",
    )
    parser.add_argument(
        "--trials-per-sg",
        type=int,
        default=5,
        help="Number of structure trials per space group",
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
        "--skip-monoclinic",
        action="store_true",
        help="Skip monoclinic space groups (focus on higher symmetry)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--save-to-db",
        action="store_true",
        help="Save structures that beat P1 to the database",
    )
    args = parser.parse_args()

    # Initialize database
    db_path = Path(args.db)
    if not db_path.exists():
        ggen_dir = Path(__file__).parent.parent.parent
        db_path = ggen_dir / args.db

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")

    logger.info(f"Using database: {db_path}")
    db = StructureDatabase(db_path)

    # Initialize calculator
    logger.info("Initializing ORB calculator...")
    calculator = get_orb_calculator()

    # Get P1 ground states
    logger.info(f"Finding P1 ground states in {args.chemsys}...")
    p1_ground_states = get_p1_ground_states(db, args.chemsys)

    if not p1_ground_states:
        logger.warning("No P1 ground states found!")
        db.close()
        return

    # Sort by energy and limit
    p1_ground_states.sort(key=lambda x: x[1])
    if len(p1_ground_states) > args.max_formulas:
        p1_ground_states = p1_ground_states[: args.max_formulas]
        logger.info(f"Limited to {args.max_formulas} lowest-energy P1 formulas")

    # Challenge each
    logger.info(f"\nChallenging {len(p1_ground_states)} P1 ground states...")
    logger.info(f"Max SGs to try: {args.max_sg}")
    logger.info(f"Trials per SG: {args.trials_per_sg}")

    results = []
    for formula, energy, struct_id in tqdm(p1_ground_states, desc="Challenging P1s"):
        result = challenge_p1_formula(
            formula=formula,
            p1_energy=energy,
            p1_id=struct_id,
            calculator=calculator,
            max_sg_to_try=args.max_sg,
            trials_per_sg=args.trials_per_sg,
            max_relax_steps=args.max_relax_steps,
            fmax=args.fmax,
            skip_monoclinic=args.skip_monoclinic,
        )
        results.append(result)

        # Save to DB if requested and P1 was beaten
        if args.save_to_db and result.p1_beaten and result.best_challenger_energy:
            # We'd need to keep the structure around - for now just log
            logger.info(
                f"Would save: {formula} at {result.best_challenger_energy:.4f} eV/atom "
                f"(SG {result.best_challenger_symbol})"
            )

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        chemsys_str = args.chemsys.replace("-", "_")
        output_path = Path(f"p1_challenge_{chemsys_str}_{ts}.json")

    # Convert to serializable format
    def result_to_dict(r: P1ChallengeResult) -> dict:
        d = asdict(r)
        # Convert sg_results
        d["sg_results"] = [asdict(sg) for sg in r.sg_results]
        return d

    output_data = {
        "parameters": {
            "database": str(db_path),
            "chemsys": args.chemsys,
            "max_formulas": args.max_formulas,
            "max_sg": args.max_sg,
            "trials_per_sg": args.trials_per_sg,
            "max_relax_steps": args.max_relax_steps,
            "fmax": args.fmax,
            "skip_monoclinic": args.skip_monoclinic,
        },
        "summary": {
            "total_challenged": len(results),
            "p1_beaten": sum(1 for r in results if r.p1_beaten),
            "avg_improvement_mev": (
                np.mean(
                    [
                        r.improvement * 1000
                        for r in results
                        if r.improvement and r.p1_beaten
                    ]
                )
                if any(r.p1_beaten for r in results)
                else 0
            ),
        },
        "results": [result_to_dict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    db.close()


if __name__ == "__main__":
    main()
