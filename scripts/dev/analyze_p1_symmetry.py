#!/usr/bin/env python3
"""
Analyze low-energy P1 structures for hidden higher symmetry.

Hypothesis: Many P1 structures from relaxations may have near-higher-symmetry
that got slightly broken. By detecting and enforcing this symmetry, we might
find even lower energy structures.

Approach:
1. Query low-energy P1 structures from the database
2. Try symmetry detection with progressively looser tolerances
3. "Idealize" the structure by snapping to detected higher symmetry
4. Re-relax (with and without symmetry constraints)
5. Compare energies to see if we found lower energy configurations

Run with: python scripts/dev/analyze_p1_symmetry.py --chemsys Bi-Fe-S
"""

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

from ggen.calculator import get_orb_calculator
from ggen.database import StructureDatabase, StoredStructure

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SymmetryAnalysisResult:
    """Result of analyzing a single P1 structure for hidden symmetry."""

    # Original structure info
    structure_id: str
    formula: str
    original_energy_per_atom: float
    original_sg_number: int
    original_sg_symbol: str

    # Detected symmetry at various tolerances
    detected_symmetries: Dict[float, Tuple[int, str]] = field(default_factory=dict)

    # Best detected higher symmetry
    best_detected_sg_number: int = 1
    best_detected_sg_symbol: str = "P1"
    detection_tolerance: float = 0.0

    # Re-relaxation results
    # Unconstrained re-relax of idealized structure
    idealized_energy_per_atom: Optional[float] = None
    idealized_final_sg_number: Optional[int] = None
    idealized_final_sg_symbol: Optional[str] = None
    idealized_relax_steps: Optional[int] = None

    # Symmetry-constrained re-relax
    constrained_energy_per_atom: Optional[float] = None
    constrained_final_sg_number: Optional[int] = None
    constrained_final_sg_symbol: Optional[str] = None
    constrained_relax_steps: Optional[int] = None

    # Energy deltas (negative = improvement)
    delta_idealized: Optional[float] = None  # idealized - original
    delta_constrained: Optional[float] = None  # constrained - original

    # Best result
    best_energy_per_atom: Optional[float] = None
    best_method: Optional[str] = None  # "original", "idealized", "constrained"

    error_message: Optional[str] = None


def detect_symmetry_sweep(
    structure: Structure,
    tolerances: List[float] = None,
) -> Dict[float, Tuple[int, str]]:
    """
    Sweep through tolerances to detect symmetry at each level.

    Returns:
        Dict mapping tolerance -> (sg_number, sg_symbol)
    """
    if tolerances is None:
        tolerances = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    results = {}
    for tol in tolerances:
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=tol)
            sg_number = analyzer.get_space_group_number()
            sg_symbol = analyzer.get_space_group_symbol()
            results[tol] = (sg_number, sg_symbol)
        except Exception as e:
            logger.debug(f"Symmetry detection failed at tol={tol}: {e}")
            results[tol] = (1, "P1")  # Fallback

    return results


def idealize_structure(
    structure: Structure,
    target_symprec: float = 0.1,
) -> Tuple[Optional[Structure], int, str]:
    """
    Idealize structure to detected symmetry at given tolerance.

    Returns:
        (idealized_structure, sg_number, sg_symbol) or (None, 1, "P1") on failure
    """
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=target_symprec)
        sg_number = analyzer.get_space_group_number()
        sg_symbol = analyzer.get_space_group_symbol()

        # Get the refined structure (snapped to ideal symmetry)
        idealized = analyzer.get_refined_structure()

        return idealized, sg_number, sg_symbol
    except Exception as e:
        logger.debug(f"Idealization failed at symprec={target_symprec}: {e}")
        return None, 1, "P1"


def relax_structure(
    structure: Structure,
    calculator,
    max_steps: int = 200,
    fmax: float = 0.02,
    preserve_symmetry: bool = False,
    symmetry_symprec: float = 0.01,
) -> Tuple[Structure, float, int, int, str]:
    """
    Relax a structure with optional symmetry preservation.

    Returns:
        (relaxed_structure, energy_per_atom, num_steps, final_sg_number, final_sg_symbol)
    """
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)
    atoms.calc = calculator

    if preserve_symmetry:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="FixSymmetry adjust_cell may be ill behaved",
                    category=UserWarning,
                )
                sym_constraint = FixSymmetry(
                    atoms,
                    symprec=symmetry_symprec,
                    adjust_positions=True,
                    adjust_cell=True,
                )
            atoms.set_constraint(sym_constraint)
        except Exception as e:
            logger.warning(f"Failed to apply symmetry constraint: {e}")

    filtered = FrechetCellFilter(atoms)
    opt = LBFGS(filtered, logfile=None, maxstep=0.2)
    opt.run(fmax=fmax, steps=max_steps)

    final_energy = atoms.get_potential_energy()
    final_energy_per_atom = final_energy / len(atoms)
    num_steps = opt.nsteps

    # Convert back and get final symmetry
    relaxed_structure = adaptor.get_structure(atoms)

    try:
        final_analyzer = SpacegroupAnalyzer(relaxed_structure, symprec=0.1)
        final_sg_number = final_analyzer.get_space_group_number()
        final_sg_symbol = final_analyzer.get_space_group_symbol()
    except Exception:
        final_sg_number = 1
        final_sg_symbol = "P1"

    return relaxed_structure, final_energy_per_atom, num_steps, final_sg_number, final_sg_symbol


def analyze_p1_structure(
    stored: StoredStructure,
    calculator,
    tolerances: List[float] = None,
    max_relax_steps: int = 200,
    fmax: float = 0.02,
) -> SymmetryAnalysisResult:
    """
    Analyze a single P1 structure for hidden symmetry and potential energy improvement.
    """
    result = SymmetryAnalysisResult(
        structure_id=stored.id,
        formula=stored.formula,
        original_energy_per_atom=stored.energy_per_atom,
        original_sg_number=stored.space_group_number or 1,
        original_sg_symbol=stored.space_group_symbol or "P1",
    )

    # Load structure
    structure = stored.get_structure()
    if structure is None:
        result.error_message = "Could not load structure from CIF"
        return result

    # Step 1: Sweep symmetry detection across tolerances
    result.detected_symmetries = detect_symmetry_sweep(structure, tolerances)

    # Find best (highest) detected symmetry
    for tol, (sg_num, sg_sym) in result.detected_symmetries.items():
        if sg_num > result.best_detected_sg_number:
            result.best_detected_sg_number = sg_num
            result.best_detected_sg_symbol = sg_sym
            result.detection_tolerance = tol

    # If no higher symmetry detected, we're done
    if result.best_detected_sg_number <= 1:
        result.best_energy_per_atom = result.original_energy_per_atom
        result.best_method = "original"
        return result

    # Step 2: Idealize the structure to the detected symmetry
    idealized, ideal_sg_num, ideal_sg_sym = idealize_structure(
        structure, target_symprec=result.detection_tolerance
    )

    if idealized is None:
        result.error_message = "Idealization failed"
        result.best_energy_per_atom = result.original_energy_per_atom
        result.best_method = "original"
        return result

    # Step 3a: Unconstrained re-relaxation of idealized structure
    try:
        _, ideal_energy, ideal_steps, ideal_final_sg, ideal_final_sym = relax_structure(
            idealized,
            calculator,
            max_steps=max_relax_steps,
            fmax=fmax,
            preserve_symmetry=False,
        )
        result.idealized_energy_per_atom = ideal_energy
        result.idealized_relax_steps = ideal_steps
        result.idealized_final_sg_number = ideal_final_sg
        result.idealized_final_sg_symbol = ideal_final_sym
        result.delta_idealized = ideal_energy - result.original_energy_per_atom
    except Exception as e:
        logger.debug(f"Unconstrained relaxation failed: {e}")
        result.idealized_energy_per_atom = None

    # Step 3b: Symmetry-constrained re-relaxation
    try:
        _, constrained_energy, constrained_steps, constrained_final_sg, constrained_final_sym = relax_structure(
            idealized,
            calculator,
            max_steps=max_relax_steps,
            fmax=fmax,
            preserve_symmetry=True,
            symmetry_symprec=0.01,
        )
        result.constrained_energy_per_atom = constrained_energy
        result.constrained_relax_steps = constrained_steps
        result.constrained_final_sg_number = constrained_final_sg
        result.constrained_final_sg_symbol = constrained_final_sym
        result.delta_constrained = constrained_energy - result.original_energy_per_atom
    except Exception as e:
        logger.debug(f"Constrained relaxation failed: {e}")
        result.constrained_energy_per_atom = None

    # Determine best result
    candidates = [("original", result.original_energy_per_atom)]
    if result.idealized_energy_per_atom is not None:
        candidates.append(("idealized", result.idealized_energy_per_atom))
    if result.constrained_energy_per_atom is not None:
        candidates.append(("constrained", result.constrained_energy_per_atom))

    best_method, best_energy = min(candidates, key=lambda x: x[1])
    result.best_method = best_method
    result.best_energy_per_atom = best_energy

    return result


def get_p1_structures(
    db: StructureDatabase,
    chemsys: Optional[str] = None,
    max_e_above_hull: float = 0.1,
    min_structures: int = 10,
    max_structures: int = 100,
) -> List[StoredStructure]:
    """
    Get low-energy P1 structures from the database.

    If chemsys is None, gets from all systems.
    """
    # Query P1 structures
    if chemsys:
        all_structures = db.get_structures_for_subsystem(chemsys, valid_only=True)
    else:
        # Get all structures
        rows = db.conn.execute(
            """
            SELECT * FROM structures 
            WHERE is_valid = 1 
            ORDER BY energy_per_atom ASC
            """
        ).fetchall()
        all_structures = [db._row_to_structure(row) for row in rows]

    # Filter to P1 only
    p1_structures = [
        s for s in all_structures
        if s.space_group_number == 1 or s.space_group_symbol == "P1"
    ]

    logger.info(f"Found {len(p1_structures)} P1 structures in database")

    # If we have hull info, filter by e_above_hull
    if chemsys and max_e_above_hull is not None:
        # Compute hull first
        db.compute_hull(chemsys)
        hull_entries = db.get_hull_entries(chemsys, e_above_hull_cutoff=max_e_above_hull)
        hull_ids = {e.id for e in hull_entries}

        p1_structures_near_hull = [
            s for s in p1_structures
            if s.id in hull_ids
        ]

        if len(p1_structures_near_hull) >= min_structures:
            p1_structures = p1_structures_near_hull
            logger.info(f"Filtered to {len(p1_structures)} P1 structures within {max_e_above_hull} eV/atom of hull")

    # Sort by energy and limit
    p1_structures.sort(key=lambda x: x.energy_per_atom)
    if max_structures and len(p1_structures) > max_structures:
        p1_structures = p1_structures[:max_structures]
        logger.info(f"Limited to {max_structures} lowest-energy P1 structures")

    return p1_structures


def print_summary(results: List[SymmetryAnalysisResult]):
    """Print a summary of the analysis results."""
    print("\n" + "=" * 80)
    print("P1 HIDDEN SYMMETRY ANALYSIS - SUMMARY")
    print("=" * 80)

    total = len(results)
    errors = sum(1 for r in results if r.error_message)
    valid = [r for r in results if not r.error_message]

    print(f"\nTotal P1 structures analyzed: {total}")
    print(f"  Errors: {errors}")
    print(f"  Valid analyses: {len(valid)}")

    if not valid:
        print("\nNo valid results to analyze.")
        return

    # How many had hidden symmetry detected?
    hidden_symmetry = [r for r in valid if r.best_detected_sg_number > 1]
    print(f"\nHidden symmetry detected: {len(hidden_symmetry)} / {len(valid)} ({100*len(hidden_symmetry)/len(valid):.1f}%)")

    if hidden_symmetry:
        # Distribution of detected space groups
        sg_counts: Dict[str, int] = {}
        for r in hidden_symmetry:
            key = f"{r.best_detected_sg_symbol} (#{r.best_detected_sg_number})"
            sg_counts[key] = sg_counts.get(key, 0) + 1

        print("\nDetected space group distribution:")
        for sg, count in sorted(sg_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {sg}: {count}")

        # Detection tolerance distribution
        tol_counts: Dict[float, int] = {}
        for r in hidden_symmetry:
            tol_counts[r.detection_tolerance] = tol_counts.get(r.detection_tolerance, 0) + 1

        print("\nDetection tolerance distribution:")
        for tol, count in sorted(tol_counts.items()):
            print(f"  symprec={tol}: {count}")

    # Energy improvement analysis
    print("\n" + "-" * 40)
    print("ENERGY IMPROVEMENT ANALYSIS")
    print("-" * 40)

    # Idealized (unconstrained) results
    idealized_results = [r for r in valid if r.delta_idealized is not None]
    if idealized_results:
        improvements = [r for r in idealized_results if r.delta_idealized < -0.001]
        avg_delta = np.mean([r.delta_idealized for r in idealized_results])
        best_improvement = min(r.delta_idealized for r in idealized_results)

        print(f"\nIdealized + Unconstrained Relax:")
        print(f"  Results: {len(idealized_results)}")
        print(f"  Improved: {len(improvements)} ({100*len(improvements)/len(idealized_results):.1f}%)")
        print(f"  Avg ΔE: {avg_delta*1000:.1f} meV/atom")
        print(f"  Best improvement: {best_improvement*1000:.1f} meV/atom")

        # Did symmetry survive unconstrained relaxation?
        symmetry_preserved = sum(
            1 for r in idealized_results
            if r.idealized_final_sg_number and r.idealized_final_sg_number > 1
        )
        print(f"  Symmetry preserved after relax: {symmetry_preserved} ({100*symmetry_preserved/len(idealized_results):.1f}%)")

    # Constrained results
    constrained_results = [r for r in valid if r.delta_constrained is not None]
    if constrained_results:
        improvements = [r for r in constrained_results if r.delta_constrained < -0.001]
        avg_delta = np.mean([r.delta_constrained for r in constrained_results])
        best_improvement = min(r.delta_constrained for r in constrained_results)

        print(f"\nIdealized + Symmetry-Constrained Relax:")
        print(f"  Results: {len(constrained_results)}")
        print(f"  Improved: {len(improvements)} ({100*len(improvements)/len(constrained_results):.1f}%)")
        print(f"  Avg ΔE: {avg_delta*1000:.1f} meV/atom")
        print(f"  Best improvement: {best_improvement*1000:.1f} meV/atom")

        # Did symmetry survive?
        symmetry_preserved = sum(
            1 for r in constrained_results
            if r.constrained_final_sg_number and r.constrained_final_sg_number > 1
        )
        print(f"  Symmetry preserved after relax: {symmetry_preserved} ({100*symmetry_preserved/len(constrained_results):.1f}%)")

    # Overall best method distribution
    print("\n" + "-" * 40)
    print("BEST METHOD DISTRIBUTION")
    print("-" * 40)

    method_counts = {"original": 0, "idealized": 0, "constrained": 0}
    for r in valid:
        if r.best_method:
            method_counts[r.best_method] = method_counts.get(r.best_method, 0) + 1

    for method, count in method_counts.items():
        pct = 100 * count / len(valid) if valid else 0
        print(f"  {method}: {count} ({pct:.1f}%)")

    # Top improvements
    print("\n" + "-" * 40)
    print("TOP 10 IMPROVEMENTS")
    print("-" * 40)

    # Get structures where symmetrization helped
    improved = [r for r in valid if r.best_method != "original" and r.best_energy_per_atom is not None]
    improved.sort(key=lambda r: (r.best_energy_per_atom - r.original_energy_per_atom))

    if improved:
        print(f"\n{'Formula':<15} {'Orig E':<12} {'New E':<12} {'ΔE (meV)':<10} {'Method':<12} {'New SG':<15}")
        print("-" * 80)
        for r in improved[:10]:
            delta = (r.best_energy_per_atom - r.original_energy_per_atom) * 1000
            new_sg = f"{r.best_detected_sg_symbol} (#{r.best_detected_sg_number})"
            print(f"{r.formula:<15} {r.original_energy_per_atom:<12.4f} {r.best_energy_per_atom:<12.4f} {delta:<10.1f} {r.best_method:<12} {new_sg:<15}")
    else:
        print("No improvements found.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze P1 structures for hidden higher symmetry"
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
        default=None,
        help="Chemical system to analyze (e.g., 'Bi-Fe-S'). If not specified, analyzes all.",
    )
    parser.add_argument(
        "--max-e-above-hull",
        type=float,
        default=0.1,
        help="Maximum e_above_hull to consider (eV/atom)",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=50,
        help="Maximum number of P1 structures to analyze",
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
        "--tolerances",
        type=str,
        default="0.001,0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5",
        help="Comma-separated list of symmetry tolerances to try",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for detailed results",
    )
    args = parser.parse_args()

    # Parse tolerances
    tolerances = [float(t) for t in args.tolerances.split(",")]

    # Initialize database
    db_path = Path(args.db)
    if not db_path.exists():
        # Try relative to ggen directory
        ggen_dir = Path(__file__).parent.parent.parent
        db_path = ggen_dir / args.db

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")

    logger.info(f"Using database: {db_path}")
    db = StructureDatabase(db_path)

    # Initialize calculator
    logger.info("Initializing ORB calculator...")
    calculator = get_orb_calculator()

    # Get P1 structures
    logger.info(f"Querying P1 structures (chemsys={args.chemsys})...")
    p1_structures = get_p1_structures(
        db,
        chemsys=args.chemsys,
        max_e_above_hull=args.max_e_above_hull,
        max_structures=args.max_structures,
    )

    if not p1_structures:
        logger.warning("No P1 structures found!")
        return

    logger.info(f"\nAnalyzing {len(p1_structures)} P1 structures...")
    logger.info(f"Tolerances: {tolerances}")
    logger.info(f"Max relax steps: {args.max_relax_steps}")
    logger.info(f"fmax: {args.fmax}")

    # Analyze each structure
    results = []
    for stored in tqdm(p1_structures, desc="Analyzing P1 structures"):
        result = analyze_p1_structure(
            stored,
            calculator,
            tolerances=tolerances,
            max_relax_steps=args.max_relax_steps,
            fmax=args.fmax,
        )
        results.append(result)

    # Print summary
    print_summary(results)

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        chemsys_str = args.chemsys.replace("-", "_") if args.chemsys else "all"
        output_path = Path(f"p1_symmetry_analysis_{chemsys_str}_{ts}.json")

    output_data = {
        "parameters": {
            "database": str(db_path),
            "chemsys": args.chemsys,
            "max_e_above_hull": args.max_e_above_hull,
            "max_structures": args.max_structures,
            "max_relax_steps": args.max_relax_steps,
            "fmax": args.fmax,
            "tolerances": tolerances,
        },
        "summary": {
            "total_analyzed": len(results),
            "errors": sum(1 for r in results if r.error_message),
            "hidden_symmetry_found": sum(1 for r in results if r.best_detected_sg_number > 1),
            "improved_by_idealization": sum(
                1 for r in results
                if r.delta_idealized is not None and r.delta_idealized < -0.001
            ),
            "improved_by_constrained": sum(
                1 for r in results
                if r.delta_constrained is not None and r.delta_constrained < -0.001
            ),
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nDetailed results saved to {output_path}")

    db.close()


if __name__ == "__main__":
    main()
