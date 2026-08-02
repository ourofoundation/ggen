"""Systematic chemical system screening.

Runs shallow explorations across candidate elements for a template like
"Fe-Bi-{X}", then ranks which systems are most promising based on convex
hull results. Helps answer: "which X should I explore deeply?"
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .calculator import RSS_LIMIT_MB, reset_dynamo_cache, rss_mb
from .colors import Colors
from .database import StructureDatabase
from .explorer import ChemistryExplorer, ExplorationResult
from .report import get_crystal_system
from .utils import normalize_dashes

logger = logging.getLogger(__name__)


# Exponential decay scale for weighted hull scoring (25 meV).
# On-hull = 1.0, 25 meV = 0.37, 50 meV = 0.14, 100 meV = 0.02.
HULL_WEIGHT_SCALE = 0.025

# Near-hull structures retained per scanned system.
TOP_HITS_PER_SYSTEM = 5


@dataclass
class TopHit:
    """A near-hull structure retained as evidence for a scanned system."""

    structure_id: str
    formula: str
    space_group: str
    space_group_number: Optional[int]
    crystal_system: str
    e_above_hull: float
    is_on_hull: bool
    num_atoms: int
    novelty: str = "known"


@dataclass
class SystemScore:
    """Scoring results for a single chemical system."""

    chemical_system: str
    variable_element: str
    near_hull_count: int = 0
    on_hull_count: int = 0
    target_crystal_system_hits: int = 0
    target_on_hull_count: int = 0
    distinct_target_formulas: int = 0
    weighted_target_score: float = 0.0
    # Lowest e_above_hull among target-symmetry hits, or among all near-hull
    # entries when the scan has no target crystal systems.
    best_e_hull: Optional[float] = None
    total_candidates: int = 0
    formulas_explored: int = 0
    num_stoichiometries: int = 0
    num_trials: int = 0
    structures_kept: int = 0
    new_composition_count: int = 0
    new_polymorph_count: int = 0
    total_time_seconds: float = 0.0
    top_hits: List[TopHit] = field(default_factory=list)
    run_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ScoutResult:
    """Results from a full scout scan across candidate elements."""

    template: str
    fixed_elements: List[str]
    candidates: List[str]
    scores: List[SystemScore]
    target_crystal_systems: Optional[List[str]] = None
    e_above_hull_cutoff: float = 0.150


class SystemScout:
    """Orchestrates shallow explorations across candidate elements to rank
    which chemical systems are most promising for deeper investigation."""

    def __init__(
        self,
        database: StructureDatabase,
        output_dir: Optional[Path] = None,
        random_seed: Optional[int] = None,
        calculator=None,
    ):
        self.database = database
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.random_seed = random_seed
        self.calculator = calculator

    @staticmethod
    def parse_template(template: str) -> List[str]:
        """Extract fixed elements from a template like 'Fe-Bi-{X}'.

        Returns the list of fixed element symbols.
        """
        parts = normalize_dashes(template).split("-")
        fixed = []
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                continue
            if part:
                fixed.append(part)
        return fixed

    @staticmethod
    def expand_template(template: str, element: str) -> str:
        """Substitute {X} in template with an element symbol.

        Returns a normalized chemical system string (e.g., 'Bi-Fe-Ti').
        """
        parts = normalize_dashes(template).split("-")
        elements = []
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                elements.append(element)
            elif part:
                elements.append(part)
        return "-".join(sorted(elements))

    def scan(
        self,
        template: str,
        candidates: List[str],
        num_trials: int = 5,
        max_atoms: int = 12,
        min_atoms: int = 2,
        crystal_systems: Optional[List[str]] = None,
        min_fraction: Optional[Dict[str, float]] = None,
        max_fraction: Optional[Dict[str, float]] = None,
        e_above_hull_cutoff: float = 0.150,
        num_workers: int = 1,
        optimization_max_steps: int = 400,
        optimization_optimizer: str = "fire",
        max_stoichiometries: Optional[int] = 50,
    ) -> ScoutResult:
        """Scan candidate elements by running shallow explorations and ranking results.

        Args:
            template: Chemical system template with {X} placeholder (e.g., 'Fe-Bi-{X}').
            candidates: List of element symbols to substitute for {X}.
            num_trials: Generation attempts per stoichiometry (lower = faster shallow scan).
            max_atoms: Maximum atoms per formula unit.
            min_atoms: Minimum atoms per formula unit.
            crystal_systems: Target crystal systems for scoring (e.g., ['tetragonal', 'hexagonal']).
                If provided, target_crystal_system_hits counts hull entries in these systems.
            min_fraction: Minimum element fraction constraints passed to explore().
            max_fraction: Maximum element fraction constraints passed to explore().
            e_above_hull_cutoff: Energy above hull cutoff for 'near hull' (eV/atom).
            num_workers: Parallel workers passed to each explore() call.
            optimization_max_steps: Max relaxation steps.
            optimization_optimizer: Optimizer for relaxation ('fire' or 'lbfgs').
            max_stoichiometries: Maximum stoichiometries to explore per candidate system.

        Returns:
            ScoutResult with scored and ranked systems.
        """
        C = Colors
        fixed_elements = self.parse_template(template)
        scores: List[SystemScore] = []

        print(
            f"\n{C.BOLD}Scout: scanning {len(candidates)} candidates for {template}{C.RESET}"
        )
        if crystal_systems:
            print(f"  Target crystal systems: {', '.join(crystal_systems)}")
        print(f"  Shallow trials: {num_trials}, max atoms: {max_atoms}")
        if max_stoichiometries is not None:
            print(f"  Max stoichiometries per system: {max_stoichiometries}")
        print(f"  E_hull cutoff: {e_above_hull_cutoff * 1000:.0f} meV/atom")
        print()

        for i, element in enumerate(candidates):
            chemsys = self.expand_template(template, element)
            label = f"[{i + 1}/{len(candidates)}] {chemsys} (X={element})"
            logger.info("START %s  RSS=%.0f MiB", label, rss_mb())
            print(f"{C.CYAN}{C.BOLD}{label}{C.RESET}")

            score = self._explore_and_score(
                chemsys=chemsys,
                variable_element=element,
                num_trials=num_trials,
                max_stoichiometries=max_stoichiometries,
                max_atoms=max_atoms,
                min_atoms=min_atoms,
                crystal_systems=crystal_systems,
                min_fraction=min_fraction,
                max_fraction=max_fraction,
                e_above_hull_cutoff=e_above_hull_cutoff,
                num_workers=num_workers,
                optimization_max_steps=optimization_max_steps,
                optimization_optimizer=optimization_optimizer,
            )
            scores.append(score)

            # Print quick summary for this system
            if score.error:
                print(f"  {C.RED}Error: {score.error}{C.RESET}")
            else:
                hull_str = f"on-hull: {score.on_hull_count}, near-hull: {score.near_hull_count}"
                if crystal_systems:
                    hull_str += (
                        f", tgt-on: {score.target_on_hull_count}"
                        f", uniq: {score.distinct_target_formulas}"
                    )
                    if score.best_e_hull is not None:
                        hull_str += f", best: {score.best_e_hull * 1000:.1f} meV"
                elapsed = _format_time(score.total_time_seconds)
                print(f"  {C.GREEN}{hull_str} ({elapsed}){C.RESET}")
            print()

        scores.sort(key=_sort_key)

        result = ScoutResult(
            template=template,
            fixed_elements=fixed_elements,
            candidates=candidates,
            scores=scores,
            target_crystal_systems=crystal_systems,
            e_above_hull_cutoff=e_above_hull_cutoff,
        )

        self._print_summary(result)
        return result

    def _explore_and_score(
        self,
        chemsys: str,
        variable_element: str,
        num_trials: int,
        max_stoichiometries: Optional[int],
        max_atoms: int,
        min_atoms: int,
        crystal_systems: Optional[List[str]],
        min_fraction: Optional[Dict[str, float]],
        max_fraction: Optional[Dict[str, float]],
        e_above_hull_cutoff: float,
        num_workers: int,
        optimization_max_steps: int,
        optimization_optimizer: str,
    ) -> SystemScore:
        """Run a shallow exploration and score the results."""
        score = SystemScore(
            chemical_system=chemsys,
            variable_element=variable_element,
        )

        start = time.time()
        explorer = None
        try:
            # Fresh explorer per system bounds long-run memory; the calculator
            # is shared so the ORB model is not reloaded for every candidate.
            explorer = ChemistryExplorer(
                random_seed=self.random_seed,
                output_dir=self.output_dir,
                database=self.database,
                calculator=self.calculator,
            )
            result: ExplorationResult = explorer.explore(
                chemical_system=chemsys,
                max_atoms=max_atoms,
                min_atoms=min_atoms,
                num_trials=num_trials,
                max_stoichiometries=max_stoichiometries,
                optimize=True,
                require_all_elements=True,
                skip_existing_formulas=False,
                use_unified_database=True,
                keep_structures_in_memory=False,
                show_progress=True,
                num_workers=num_workers,
                min_fraction=min_fraction,
                max_fraction=max_fraction,
                optimization_max_steps=optimization_max_steps,
                optimization_optimizer=optimization_optimizer,
            )

            score.total_candidates = result.num_successful
            score.formulas_explored = result.num_candidates
            score.num_stoichiometries = result.num_stoichiometries
            score.num_trials = result.num_trials

            # Only score structures generated in THIS run so that
            # previously-explored systems don't inflate the ranking.
            # Hull distances are still computed from all data.
            score.run_id = explorer._unified_run_id
            score.structures_kept = self.database.get_run_statistics(score.run_id)[
                "structures"
            ]

            # Terminal elements sit on the hull by construction, so they are
            # never evidence that a system is worth exploring.
            hull_entries = [
                entry
                for entry in self.database.get_hull_entries(
                    chemsys,
                    e_above_hull_cutoff=e_above_hull_cutoff,
                    run_id=score.run_id,
                )
                if len(entry.elements) > 1
            ]
            known_polymorphs = self.database.get_known_polymorphs(chemsys)

            targets = set(crystal_systems or [])
            target_formulas: set[str] = set()
            hits: List[TopHit] = []

            for entry in hull_entries:
                if entry.is_on_hull:
                    score.on_hull_count += 1
                score.near_hull_count += 1

                novelty = self.database.classify_novelty(entry, known_polymorphs)
                if novelty == "new composition":
                    score.new_composition_count += 1
                elif novelty == "new polymorph":
                    score.new_polymorph_count += 1

                e_hull = entry.e_above_hull or 0.0
                crystal_system = (
                    get_crystal_system(entry.space_group_number)
                    if entry.space_group_number is not None
                    else "unknown"
                )
                hits.append(
                    TopHit(
                        structure_id=entry.id,
                        formula=entry.formula,
                        space_group=entry.space_group_symbol or "?",
                        space_group_number=entry.space_group_number,
                        crystal_system=crystal_system,
                        e_above_hull=e_hull,
                        is_on_hull=entry.is_on_hull,
                        num_atoms=entry.num_atoms,
                        novelty=novelty,
                    )
                )

                if crystal_system in targets:
                    score.target_crystal_system_hits += 1
                    if entry.is_on_hull:
                        score.target_on_hull_count += 1
                    target_formulas.add(entry.formula)
                    score.weighted_target_score += math.exp(-e_hull / HULL_WEIGHT_SCALE)

            score.distinct_target_formulas = len(target_formulas)

            scored_hits = (
                [h for h in hits if h.crystal_system in targets] if targets else hits
            )
            score.best_e_hull = min((h.e_above_hull for h in scored_hits), default=None)

            # Target-symmetry hits first, then the closest remaining phases, so
            # a scan without target systems still returns its best structures.
            hits.sort(key=lambda h: (h.crystal_system not in targets, h.e_above_hull))
            score.top_hits = hits[:TOP_HITS_PER_SYSTEM]

        except Exception as e:
            logger.error("Exploration failed for %s: %s", chemsys, e)
            score.error = str(e)
        finally:
            explorer = None
            gc.collect()

        current_rss = rss_mb()
        if current_rss > RSS_LIMIT_MB:
            logger.warning(
                "RSS %.0f MiB exceeds limit %d MiB; clearing compile and CUDA caches",
                current_rss,
                RSS_LIMIT_MB,
            )
            reset_dynamo_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_rss = rss_mb()
        logger.info("RSS after %s: %.0f MiB", chemsys, current_rss)

        score.total_time_seconds = time.time() - start
        return score

    def _print_summary(self, result: ScoutResult) -> None:
        """Print a ranked summary table with optional top-hits detail."""
        C = Colors
        scores = result.scores
        has_targets = result.target_crystal_systems is not None

        print(f"\n{C.BOLD}{'=' * 80}{C.RESET}")
        print(f"{C.BOLD}Scout Results: {result.template}{C.RESET}")
        if result.target_crystal_systems:
            print(f"Target crystal systems: {', '.join(result.target_crystal_systems)}")
        print(f"E_hull cutoff: {result.e_above_hull_cutoff * 1000:.0f} meV/atom")
        print(f"{C.BOLD}{'=' * 80}{C.RESET}\n")

        # Header
        header = f"{'Rank':<5} {'System':<16} {'X':<5}"
        if has_targets:
            header += f" {'Tgt-On':>6} {'Uniq':>5} {'Score':>7}"
        header += (
            f" {'Best-E_hull':>12} {'On-Hull':>8} {'Near-Hull':>10}"
            f" {'New':>5} {'Time':>8}"
        )
        print(f"{C.BOLD}{header}{C.RESET}")
        print("-" * len(header.replace(C.BOLD, "").replace(C.RESET, "")))

        for rank, score in enumerate(scores, 1):
            if score.error:
                line = f"{rank:<5} {score.chemical_system:<16} {score.variable_element:<5} {'ERROR':>8}"
                print(f"{C.RED}{line}  {score.error}{C.RESET}")
                continue

            elapsed = _format_time(score.total_time_seconds)

            e_hull_str = (
                f"{score.best_e_hull * 1000:.1f} meV"
                if score.best_e_hull is not None
                else "-"
            )

            line = f"{rank:<5} {score.chemical_system:<16} {score.variable_element:<5}"
            if has_targets:
                line += (
                    f" {score.target_on_hull_count:>6}"
                    f" {score.distinct_target_formulas:>5}"
                    f" {score.weighted_target_score:>7.1f}"
                )
            line += (
                f" {e_hull_str:>12}"
                f" {score.on_hull_count:>8}"
                f" {score.near_hull_count:>10}"
                f" {score.new_composition_count + score.new_polymorph_count:>5}"
                f" {elapsed:>8}"
            )

            if rank <= 3 and score.near_hull_count > 0:
                print(f"{C.GREEN}{line}{C.RESET}")
            else:
                print(line)

        # Best near-hull structure detail
        if any(s.top_hits for s in scores if not s.error):
            print(f"\n{C.BOLD}Top hits:{C.RESET}")
            for rank, score in enumerate(scores, 1):
                if score.error or not score.top_hits:
                    continue
                hits_str = ", ".join(
                    f"{h.formula} / {h.space_group} ({h.e_above_hull * 1000:.0f} meV)"
                    for h in score.top_hits
                )
                print(f"  {rank}. {C.CYAN}{score.chemical_system}{C.RESET}: {hits_str}")

        print()


def _sort_key(score: SystemScore):
    """Sort key for ranking systems.

    Priority: stable phases in target symmetry, compositional breadth, then
    quality-weighted target score. Hull counts and best E_hull break the
    remaining ties, and are what rank a scan with no target crystal systems.
    """
    if score.error:
        return (0, 0, 0.0, 0, 0, float("inf"))
    best_e = score.best_e_hull if score.best_e_hull is not None else float("inf")
    return (
        -score.target_on_hull_count,
        -score.distinct_target_formulas,
        -score.weighted_target_score,
        -score.on_hull_count,
        -score.near_hull_count,
        best_e,
    )


def _format_time(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"
