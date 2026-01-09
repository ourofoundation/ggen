#!/usr/bin/env python3
"""
Re-relax structures in the database using torch-sim batched GPU relaxation.

Useful for fixing structures that may have been insufficiently relaxed
(e.g., due to torch-sim FIRE not using force convergence criterion).

Usage:
    # Re-relax unstable structures within 150 meV of hull
    python relax.py --unstable-only

    # Specific chemical system
    python relax.py --system Fe-Mn-Co --unstable-only

    # Custom batch size for GPU memory management
    python relax.py --unstable-only --batch-size 32

    # Re-relax ALL structures near hull (use with caution)
    python relax.py --e-above-hull 0.05

    # Sequential relaxation (no batching, for debugging)
    python relax.py --unstable-only --sequential
"""

import argparse
import logging
import sys
import warnings
from typing import Any, Dict, List, Optional

import torch
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ggen import Colors, StructureDatabase
from ggen.database import StoredStructure

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=UserWarning, module="orb_models")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_entries_for_relaxation(
    db: StructureDatabase,
    chemical_system: Optional[str] = None,
    e_above_hull_cutoff: float = 0.15,
    max_structures: Optional[int] = None,
    unstable_only: bool = False,
    all_structures: bool = False,
) -> List[StoredStructure]:
    """
    Get database entries that should be re-relaxed.

    Args:
        db: Database connection
        chemical_system: Optional specific system (e.g., "Fe-Mn-Co"). If None, all systems.
        e_above_hull_cutoff: Maximum energy above hull in eV/atom
        max_structures: Optional limit on number of structures
        unstable_only: If True, only get structures marked as dynamically unstable
        all_structures: If True, get all structures (ignores e_above_hull_cutoff)

    Returns:
        List of StoredStructure objects to re-relax
    """
    conn = db.conn

    # Build WHERE clauses
    where_clauses = ["s.cif_content IS NOT NULL"]
    params: List = []

    if chemical_system:
        chemsys = db.normalize_chemsys(chemical_system)
        where_clauses.append("h.chemsys = ?")
        params.append(chemsys)

    if unstable_only:
        where_clauses.append("s.is_dynamically_stable = 0")

    if not all_structures:
        where_clauses.append("h.e_above_hull <= ?")
        params.append(e_above_hull_cutoff)

    where_sql = " AND ".join(where_clauses)

    if chemical_system:
        query = f"""
            SELECT s.*, h.e_above_hull, h.is_on_hull
            FROM structures s
            JOIN hull_entries h ON s.id = h.structure_id
            WHERE {where_sql}
            ORDER BY h.e_above_hull ASC, s.energy_per_atom ASC
        """
    else:
        query = f"""
            SELECT DISTINCT s.*, MIN(h.e_above_hull) as e_above_hull, MAX(h.is_on_hull) as is_on_hull
            FROM structures s
            JOIN hull_entries h ON s.id = h.structure_id
            WHERE {where_sql}
            GROUP BY s.id
            ORDER BY e_above_hull ASC, s.energy_per_atom ASC
        """

    if max_structures:
        query += f" LIMIT {max_structures}"

    rows = conn.execute(query, params).fetchall()

    structures = []
    for row in rows:
        s = db._row_to_structure(row)
        s.e_above_hull = row["e_above_hull"]
        s.is_on_hull = bool(row["is_on_hull"])
        structures.append(s)

    return structures


def relax_batch_torchsim(
    entries: List[StoredStructure],
    calculator,
    max_steps: int = 200,
    fmax: float = 0.02,
) -> List[Dict[str, Any]]:
    """
    Relax a batch of structures using torch-sim GPU batched optimization.

    Returns list of result dicts (one per entry).
    """
    import torch_sim as ts
    from torch_sim.models.orb import OrbModel as TorchSimOrbModel

    adaptor = AseAtomsAdaptor()
    results = []

    # Convert entries to atoms, tracking which ones succeeded
    atoms_list = []
    entry_indices = []  # Maps atoms_list index -> entries index
    initial_energies = []

    for i, entry in enumerate(entries):
        pymatgen_structure = entry.get_structure()
        if pymatgen_structure is None:
            results.append({"error": "Could not load structure from CIF", "index": i})
            continue

        try:
            atoms = adaptor.get_atoms(pymatgen_structure)
            atoms.calc = calculator
            initial_energy = atoms.get_potential_energy()
            initial_energies.append(initial_energy / len(atoms))
            atoms_list.append(atoms)
            entry_indices.append(i)
        except Exception as e:
            results.append({"error": str(e), "index": i})

    if not atoms_list:
        return results

    # Get the raw ORB model from calculator
    if hasattr(calculator, "orbff"):
        raw_model = calculator.orbff
    elif hasattr(calculator, "model"):
        raw_model = calculator.model
    else:
        # Fallback to sequential if we can't get the model
        raise AttributeError("Cannot extract ORB model from calculator")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create torch-sim model wrapper
    ts_model = TorchSimOrbModel(
        model=raw_model,
        compute_stress=True,
        compute_forces=True,
        device=device,
    )

    # Run batched optimization with force convergence
    try:
        final_state = ts.optimize(
            system=atoms_list,
            model=ts_model,
            optimizer=ts.Optimizer.fire,
            max_steps=max_steps,
            autobatcher=False,
            init_kwargs={"cell_filter": ts.CellFilter.frechet},
            convergence_fn=ts.generate_force_convergence_fn(force_tol=fmax),
            pbar=True,
        )

        # Extract results
        final_atoms_list = final_state.to_atoms()
        energies = final_state.energy.detach().cpu().numpy()

        for j, atoms in enumerate(final_atoms_list):
            entry_idx = entry_indices[j]
            entry = entries[entry_idx]

            try:
                final_energy = float(energies[j])
                final_energy_per_atom = final_energy / len(atoms)
                initial_e = initial_energies[j]

                # Convert back to pymatgen Structure
                relaxed_structure = adaptor.get_structure(atoms)

                # Get space group info
                try:
                    analyzer = SpacegroupAnalyzer(relaxed_structure, symprec=0.1)
                    sg_number = analyzer.get_space_group_number()
                    sg_symbol = analyzer.get_space_group_symbol()
                except Exception:
                    sg_number = entry.space_group_number
                    sg_symbol = entry.space_group_symbol

                # Generate CIF
                cif_writer = CifWriter(relaxed_structure, symprec=0.1)
                cif_content = str(cif_writer)

                results.append(
                    {
                        "index": entry_idx,
                        "success": True,
                        "initial_energy_per_atom": initial_e,
                        "final_energy_per_atom": final_energy_per_atom,
                        "final_total_energy": final_energy,
                        "energy_change": final_energy_per_atom - initial_e,
                        "num_steps": max_steps,  # torch-sim doesn't track per-structure
                        "cif_content": cif_content,
                        "space_group_number": sg_number,
                        "space_group_symbol": sg_symbol,
                        "converged": True,
                    }
                )
            except Exception as e:
                results.append({"error": str(e), "index": entry_idx})

    except Exception as e:
        # Mark all as failed
        for j, entry_idx in enumerate(entry_indices):
            results.append(
                {"error": f"Batch relaxation failed: {e}", "index": entry_idx}
            )

    return results


def relax_structure_sequential(
    structure: StoredStructure,
    calculator,
    max_steps: int = 200,
    fmax: float = 0.02,
) -> dict:
    """
    Re-relax a structure using ASE LBFGS (sequential fallback).

    Returns dict with relaxation results or error info.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    pymatgen_structure = structure.get_structure()
    if pymatgen_structure is None:
        return {"error": "Could not load structure from CIF"}

    try:
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(pymatgen_structure)
        atoms.calc = calculator

        # Get initial energy
        initial_energy = atoms.get_potential_energy()
        initial_energy_per_atom = initial_energy / len(atoms)

        # Relax with LBFGS
        filtered = FrechetCellFilter(atoms)
        opt = LBFGS(filtered, logfile=None, maxstep=0.2)
        opt.run(fmax=fmax, steps=max_steps)

        # Get final energy
        final_energy = atoms.get_potential_energy()
        final_energy_per_atom = final_energy / len(atoms)
        num_steps = opt.nsteps

        # Convert back to pymatgen Structure
        relaxed_structure = adaptor.get_structure(atoms)

        # Get space group info
        try:
            analyzer = SpacegroupAnalyzer(relaxed_structure, symprec=0.1)
            sg_number = analyzer.get_space_group_number()
            sg_symbol = analyzer.get_space_group_symbol()
        except Exception:
            sg_number = structure.space_group_number
            sg_symbol = structure.space_group_symbol

        # Generate CIF
        cif_writer = CifWriter(relaxed_structure, symprec=0.1)
        cif_content = str(cif_writer)

        return {
            "success": True,
            "initial_energy_per_atom": initial_energy_per_atom,
            "final_energy_per_atom": final_energy_per_atom,
            "final_total_energy": final_energy,
            "energy_change": final_energy_per_atom - initial_energy_per_atom,
            "num_steps": num_steps,
            "cif_content": cif_content,
            "space_group_number": sg_number,
            "space_group_symbol": sg_symbol,
            "converged": num_steps < max_steps,
        }
    except Exception as e:
        return {"error": str(e)}


def process_result(
    entry: StoredStructure,
    result: dict,
    db: StructureDatabase,
    min_energy_change: float,
    clear_phonons: bool,
    C: Any,
) -> tuple:
    """Process a single relaxation result and update database if needed.

    Returns (improved: bool, energy_saved: float, failed: bool)
    """
    if "error" in result:
        logger.info(f"  {C.YELLOW}⚠ Failed: {result['error']}{C.RESET}")
        return False, 0.0, True

    energy_change = result["energy_change"]

    # Only update if energy improved significantly
    if energy_change < -min_energy_change:
        # Update database with new structure
        db._update_structure(
            structure_id=entry.id,
            energy_per_atom=result["final_energy_per_atom"],
            total_energy=result["final_total_energy"],
            cif_content=result["cif_content"],
            clear_phonon_data=clear_phonons,
        )

        logger.info(
            f"  {C.GREEN}✓ Improved{C.RESET} "
            f"ΔE={energy_change * 1000:.1f} meV/atom "
            f"({entry.energy_per_atom:.4f} → {result['final_energy_per_atom']:.4f} eV/atom)"
        )
        if result.get("space_group_symbol") != entry.space_group_symbol:
            logger.info(
                f"    Space group: {entry.space_group_symbol} → {result['space_group_symbol']}"
            )
        return True, abs(energy_change), False
    else:
        status = "converged" if result.get("converged", True) else "no improvement"
        logger.info(
            f"  {C.DIM}– {status.capitalize()}{C.RESET} "
            f"(ΔE={energy_change * 1000:+.1f} meV/atom)"
        )
        return False, 0.0, False


def main():
    parser = argparse.ArgumentParser(
        description="Re-relax structures in the database using torch-sim batched GPU relaxation"
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Chemical system to process (e.g., 'Fe-Mn-Co'). Default: all systems.",
    )
    parser.add_argument(
        "--e-above-hull",
        type=float,
        default=0.15,
        help="Max energy above hull in eV/atom (default: 0.15 = 150 meV)",
    )
    parser.add_argument(
        "--max-structures",
        type=int,
        default=None,
        help="Maximum number of structures to process",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum relaxation steps per structure (default: 400)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.02,
        help="Force convergence criterion in eV/Å (default: 0.02)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for GPU relaxation (default: 16). Reduce if OOM.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential LBFGS relaxation instead of batched torch-sim",
    )
    parser.add_argument(
        "--unstable-only",
        action="store_true",
        help="Only re-relax structures marked as dynamically unstable",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-relax ALL structures (ignores --e-above-hull)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="ggen.db",
        help="Path to database file (default: ggen.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List structures that would be processed without running relaxation",
    )
    parser.add_argument(
        "--min-energy-change",
        type=float,
        default=0.001,
        help="Minimum energy change (eV/atom) to update database (default: 0.001)",
    )
    parser.add_argument(
        "--keep-phonons",
        action="store_true",
        help="Keep existing phonon data (default: clear phonons for improved structures)",
    )
    args = parser.parse_args()

    C = Colors

    # Connect to database
    logger.info(f"{C.BOLD}Structure Re-relaxation{C.RESET}")
    logger.info(f"{C.DIM}{'=' * 50}{C.RESET}")
    logger.info(f"Database: {C.CYAN}{args.database}{C.RESET}")

    try:
        db = StructureDatabase(args.database)
    except Exception as e:
        logger.error(f"{C.RED}Failed to open database: {e}{C.RESET}")
        sys.exit(1)

    # Get structures to re-relax
    filter_desc = []
    if args.unstable_only:
        filter_desc.append("dynamically unstable")
    if not args.all:
        filter_desc.append(f"E_hull ≤ {args.e_above_hull * 1000:.0f} meV")
    filter_str = " AND ".join(filter_desc) if filter_desc else "all structures"

    logger.info(f"Finding entries: {filter_str}...")

    entries = get_entries_for_relaxation(
        db=db,
        chemical_system=args.system,
        e_above_hull_cutoff=args.e_above_hull,
        max_structures=args.max_structures,
        unstable_only=args.unstable_only,
        all_structures=args.all,
    )

    if not entries:
        logger.info(f"{C.GREEN}No entries match the criteria!{C.RESET}")
        db.close()
        return

    logger.info(f"Found {C.YELLOW}{len(entries)}{C.RESET} entries to process")
    logger.info("")

    if args.dry_run:
        logger.info(f"{C.BOLD}Dry run - would process:{C.RESET}")
        for i, s in enumerate(entries):
            e_hull = s.e_above_hull or 0
            stable_str = ""
            if s.is_dynamically_stable is not None:
                stable_str = f"  {'stable' if s.is_dynamically_stable else 'UNSTABLE'}"
            logger.info(
                f"  {i + 1:3d}. {s.formula:12s}  "
                f"E={s.energy_per_atom:.4f} eV/atom  "
                f"SG={s.space_group_symbol:10s}  "
                f"E_hull={e_hull * 1000:.1f} meV{stable_str}"
            )
        db.close()
        return

    # Initialize calculator
    logger.info(f"{C.BOLD}Initializing calculator...{C.RESET}")
    try:
        from ggen.calculator import get_orb_calculator

        calculator = get_orb_calculator()
        logger.info(f"  Using: {C.CYAN}ORB v3 conservative{C.RESET}")
    except Exception as e:
        logger.error(f"{C.RED}Failed to initialize calculator: {e}{C.RESET}")
        db.close()
        sys.exit(1)

    device = "GPU" if torch.cuda.is_available() else "CPU"
    mode = "sequential LBFGS" if args.sequential else f"batched torch-sim ({device})"
    logger.info(f"  Mode: {C.CYAN}{mode}{C.RESET}")
    if not args.sequential:
        logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  fmax: {args.fmax} eV/Å")
    logger.info("")

    # Process entries
    logger.info(f"{C.BOLD}Re-relaxing structures...{C.RESET}")
    logger.info(f"{C.DIM}{'-' * 50}{C.RESET}")

    num_improved = 0
    num_unchanged = 0
    num_failed = 0
    total_energy_saved = 0.0

    if args.sequential:
        # Sequential processing
        for i, entry in enumerate(entries):
            e_hull = entry.e_above_hull or 0
            stable_str = ""
            if entry.is_dynamically_stable is not None:
                stable_str = (
                    f" ({'stable' if entry.is_dynamically_stable else 'unstable'})"
                )

            logger.info(
                f"{C.CYAN}[{i + 1}/{len(entries)}]{C.RESET} "
                f"{C.BOLD}{entry.formula:12s}{C.RESET}  "
                f"E_hull={e_hull * 1000:.1f} meV  "
                f"SG={entry.space_group_symbol}{stable_str}"
            )

            result = relax_structure_sequential(
                structure=entry,
                calculator=calculator,
                max_steps=args.max_steps,
                fmax=args.fmax,
            )

            improved, energy_saved, failed = process_result(
                entry, result, db, args.min_energy_change, not args.keep_phonons, C
            )
            if improved:
                num_improved += 1
                total_energy_saved += energy_saved
            elif failed:
                num_failed += 1
            else:
                num_unchanged += 1
    else:
        # Batched processing
        num_batches = (len(entries) + args.batch_size - 1) // args.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(entries))
            batch_entries = entries[start_idx:end_idx]

            logger.info(
                f"{C.CYAN}[Batch {batch_idx + 1}/{num_batches}]{C.RESET} "
                f"Processing {len(batch_entries)} structures..."
            )

            # Show what's in the batch
            for entry in batch_entries:
                e_hull = entry.e_above_hull or 0
                stable_str = ""
                if entry.is_dynamically_stable is not None:
                    stable_str = (
                        f" ({'stable' if entry.is_dynamically_stable else 'unstable'})"
                    )
                logger.info(
                    f"  • {entry.formula:12s} E_hull={e_hull * 1000:.1f} meV "
                    f"SG={entry.space_group_symbol}{stable_str}"
                )

            # Run batch relaxation
            results = relax_batch_torchsim(
                entries=batch_entries,
                calculator=calculator,
                max_steps=args.max_steps,
                fmax=args.fmax,
            )

            # Sort results by index to match entries order
            results_by_idx = {r.get("index", i): r for i, r in enumerate(results)}

            # Process results
            logger.info(f"{C.DIM}Results:{C.RESET}")
            for i, entry in enumerate(batch_entries):
                result = results_by_idx.get(i, {"error": "No result returned"})
                formula = f"{C.BOLD}{entry.formula:12s}{C.RESET}"

                if "error" in result:
                    logger.info(
                        f"  {formula} {C.YELLOW}⚠ Failed: {result['error']}{C.RESET}"
                    )
                    num_failed += 1
                    continue

                energy_change = result["energy_change"]

                if energy_change < -args.min_energy_change:
                    # Update database
                    db._update_structure(
                        structure_id=entry.id,
                        energy_per_atom=result["final_energy_per_atom"],
                        total_energy=result["final_total_energy"],
                        cif_content=result["cif_content"],
                        clear_phonon_data=not args.keep_phonons,
                    )

                    logger.info(
                        f"  {formula} {C.GREEN}✓ Improved{C.RESET} "
                        f"ΔE={energy_change * 1000:.1f} meV/atom "
                        f"({entry.energy_per_atom:.4f} → {result['final_energy_per_atom']:.4f})"
                    )
                    num_improved += 1
                    total_energy_saved += abs(energy_change)
                else:
                    logger.info(
                        f"  {formula} {C.DIM}– No change{C.RESET} "
                        f"(ΔE={energy_change * 1000:+.1f} meV/atom)"
                    )
                    num_unchanged += 1

            logger.info("")

    # Summary
    logger.info("")
    logger.info(f"{C.BOLD}Summary{C.RESET}")
    logger.info(f"{C.DIM}{'-' * 50}{C.RESET}")
    logger.info(f"  Processed:  {len(entries)}")
    logger.info(f"  Improved:   {C.GREEN}{num_improved}{C.RESET}")
    logger.info(f"  Unchanged:  {num_unchanged}")
    if num_failed:
        logger.info(f"  Failed:     {C.YELLOW}{num_failed}{C.RESET}")
    if num_improved > 0:
        logger.info(
            f"  Total energy saved: {C.GREEN}{total_energy_saved * 1000:.1f} meV/atom{C.RESET} "
            f"(avg {total_energy_saved / num_improved * 1000:.1f} meV/atom)"
        )

    # Reminder about hull recomputation
    if num_improved > 0:
        logger.info("")
        logger.info(
            f"{C.YELLOW}Note: Run hull recomputation to update E_above_hull values.{C.RESET}"
        )
        if not args.keep_phonons:
            logger.info(
                f"{C.YELLOW}Phonon data cleared for {num_improved} structures - "
                f"run phonons.py to recalculate.{C.RESET}"
            )

    db.close()
    logger.info("")
    logger.info(f"{C.GREEN}{C.BOLD}Done!{C.RESET}")


if __name__ == "__main__":
    main()
