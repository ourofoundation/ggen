#!/usr/bin/env python3
"""
Import external crystal datasets into the ggen database.

Supports importing from:
  - Materials Project (provider='mp')
  - Alexandria OPTIMADE (provider='alexandria')

Imported structures are stored with CIF geometry and source metadata, but
without ORB energies -- run relax.py afterwards to populate energies in a
consistent domain.

Supports resuming interrupted imports: already-imported source IDs are
automatically skipped.

Usage:
    # Import all Materials Project structures
    python import.py --database ggen.db --provider mp

    # Import Alexandria structures
    python import.py --database ggen.db --provider alexandria

    # Import Alexandria from local .json.bz2 chunk files
    python import.py --database ggen.db --provider alexandria \
        --alexandria-local-dir alexandria_pbe_2024_12_15

    # Filter to specific elements
    python import.py --database ggen.db --provider alexandria --elements Fe Mn Co

    # Dry run -- show what would be imported
    python import.py --database ggen.db --provider alexandria --dry-run

    # Import only stable structures (provider-specific definition)
    python import.py --database ggen.db --provider alexandria --stable-only

Materials Project requires MP_API_KEY environment variable or --api-key.
"""

import argparse
import bz2
import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import requests
from requests.exceptions import ConnectionError, Timeout
from tqdm import tqdm

from ggen import Colors, StructureDatabase

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_source_id(source: str, material_id: str) -> str:
    """Build a globally unique source_id key for database storage."""
    if source == "alexandria":
        return f"alexandria:{material_id}"
    return material_id


def count_already_imported(db: StructureDatabase, source: str) -> int:
    """Count structures already imported from a source."""
    row = db.conn.execute(
        "SELECT COUNT(*) FROM structures WHERE source = ? AND source_id IS NOT NULL",
        (source,),
    ).fetchone()
    return int(row[0]) if row else 0


def is_already_imported(db: StructureDatabase, source: str, material_id: str) -> bool:
    """Check if a material_id already exists in the DB."""
    source_id = build_source_id(source, material_id)
    row = db.conn.execute(
        "SELECT 1 FROM structures WHERE source = ? AND source_id = ? LIMIT 1",
        (source, source_id),
    ).fetchone()
    return row is not None


def load_existing_source_ids(db: StructureDatabase, source: str) -> set[str]:
    """Load existing source_ids for fast membership checks."""
    rows = db.conn.execute(
        "SELECT source_id FROM structures WHERE source = ? AND source_id IS NOT NULL",
        (source,),
    ).fetchall()
    return {str(row[0]) for row in rows}


def fetch_mp_summaries(
    api_key: str,
    elements: Optional[List[str]] = None,
    stable_only: bool = False,
    chunk_size: int = 1000,
) -> list:
    """
    Fetch all material summaries from Materials Project.

    Returns a list of SummaryDoc objects with structure, formula, symmetry, energy info.
    """
    from mp_api.client import MPRester

    logger.info(f"Connecting to Materials Project API...")

    with MPRester(api_key=api_key) as mpr:
        # Build query kwargs
        kwargs: Dict[str, Any] = {
            "fields": [
                "material_id",
                "structure",
                "formula_pretty",
                "symmetry",
                "energy_per_atom",
                "energy_above_hull",
                "is_stable",
                "nsites",
            ],
            "chunk_size": chunk_size,
        }

        if elements:
            kwargs["elements"] = elements

        if stable_only:
            kwargs["is_stable"] = True

        logger.info(f"Fetching material summaries...")
        if elements:
            logger.info(f"  Filtering to elements: {', '.join(elements)}")
        if stable_only:
            logger.info(f"  Filtering to stable materials only")

        docs = mpr.materials.summary.search(**kwargs)

    logger.info(f"Fetched {len(docs)} materials from MP")
    return docs


def _request_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> requests.Response:
    """
    GET with exponential backoff on transient errors (429, 5xx, timeouts).

    Reduces page_limit on 403 (Alexandria rejects oversized pages).
    """
    retries = 0
    while True:
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 403 and params is not None:
                page_limit = int(params.get("page_limit", 0))
                if page_limit > 1:
                    next_limit = max(1, page_limit // 2)
                    logger.warning(
                        f"  Received 403 with page_limit={page_limit}; "
                        f"retrying with page_limit={next_limit}"
                    )
                    params = dict(params)
                    params["page_limit"] = next_limit
                    continue
            if response.status_code in (429, 500, 502, 503, 504):
                if retries >= max_retries:
                    response.raise_for_status()
                delay = base_delay * (2 ** retries)
                logger.warning(
                    f"  HTTP {response.status_code}; retrying in {delay:.0f}s "
                    f"({retries + 1}/{max_retries})"
                )
                time.sleep(delay)
                retries += 1
                continue
            response.raise_for_status()
            return response
        except (ConnectionError, Timeout) as exc:
            if retries >= max_retries:
                raise
            delay = base_delay * (2 ** retries)
            logger.warning(
                f"  {type(exc).__name__}; retrying in {delay:.0f}s "
                f"({retries + 1}/{max_retries})"
            )
            time.sleep(delay)
            retries += 1


class AlexandriaStream:
    """
    Streaming iterator over Alexandria OPTIMADE structures.

    Fetches page-by-page so only one page is in memory at a time.
    Exposes ``total_available`` (set after the first page is received)
    and ``fetched_count`` for progress tracking.
    """

    RESPONSE_FIELDS = [
        "chemical_formula_reduced",
        "elements",
        "nsites",
        "lattice_vectors",
        "cartesian_site_positions",
        "species_at_sites",
        "species",
        "last_modified",
        "_alexandria_hull_distance",
        "_alexandria_formation_energy_per_atom",
        "_alexandria_energy",
        "_alexandria_energy_corrected",
        "_alexandria_band_gap",
        "_alexandria_space_group",
    ]

    def __init__(
        self,
        endpoint_url: str,
        elements: Optional[List[str]] = None,
        stable_only: bool = False,
        chunk_size: int = 1000,
        stable_threshold: float = 1e-8,
        timeout: int = 60,
    ):
        self.endpoint_url = endpoint_url
        self.elements = elements
        self.stable_only = stable_only
        self.stable_threshold = stable_threshold
        self.timeout = timeout

        self.page_limit = max(1, min(int(chunk_size), 500))
        if self.page_limit != chunk_size:
            logger.info(
                f"  Adjusting chunk size for Alexandria: {chunk_size} -> {self.page_limit}"
            )

        self.total_available: Optional[int] = None
        self.fetched_count: int = 0

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        logger.info("Connecting to Alexandria OPTIMADE API...")
        logger.info(f"  Endpoint: {self.endpoint_url}")

        filter_clauses: List[str] = []
        if self.elements:
            quoted = ", ".join(f'"{el}"' for el in self.elements)
            filter_clauses.append(f"elements HAS ALL {quoted}")
            logger.info(f"  Filtering to elements: {', '.join(self.elements)}")
        if self.stable_only:
            filter_clauses.append(
                f"_alexandria_hull_distance <= {self.stable_threshold}"
            )
            logger.info(
                f"  Filtering to stable only (hull distance <= {self.stable_threshold})"
            )

        params: Dict[str, Any] = {
            "page_limit": self.page_limit,
            "response_fields": ",".join(self.RESPONSE_FIELDS),
        }
        if filter_clauses:
            params["filter"] = " AND ".join(filter_clauses)

        next_url: Optional[str] = self.endpoint_url
        request_params: Optional[Dict[str, Any]] = params

        while next_url:
            response = _request_with_retry(
                next_url, params=request_params, timeout=self.timeout
            )
            payload = response.json()

            if self.total_available is None:
                meta = payload.get("meta", {})
                self.total_available = meta.get("data_available")

            page_docs = payload.get("data", [])
            self.fetched_count += len(page_docs)
            yield from page_docs

            links = payload.get("links", {})
            next_link = links.get("next")
            if isinstance(next_link, dict):
                next_url = next_link.get("href")
            elif isinstance(next_link, str):
                next_url = next_link
            else:
                next_url = None
            request_params = None  # next link already includes query params


class LocalAlexandriaStream:
    """
    Stream Alexandria entries from local .json.bz2 chunk files.

    Local chunk files contain a top-level ``entries`` list of serialized
    ComputedStructureEntry-like dicts. This stream converts each entry into the
    same lightweight doc shape consumed by ``import_alexandria_material``:
      {"id": "...", "attributes": {...}}
    """

    def __init__(
        self,
        local_dir: str,
        elements: Optional[List[str]] = None,
        stable_only: bool = False,
        stable_threshold: float = 1e-8,
        file_pattern: str = "alexandria_*.json.bz2",
    ):
        self.local_dir = Path(local_dir)
        self.elements = set(elements or [])
        self.stable_only = stable_only
        self.stable_threshold = stable_threshold
        self.file_pattern = file_pattern
        self.total_available: Optional[int] = None
        self.fetched_count: int = 0

    @staticmethod
    def _to_doc(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = entry.get("data", {})
        structure = entry.get("structure", {})
        lattice = structure.get("lattice", {})
        sites = structure.get("sites", [])

        material_id = data.get("mat_id") or entry.get("entry_id")
        lattice_vectors = lattice.get("matrix")
        if not material_id or lattice_vectors is None or not sites:
            return None

        cartesian_site_positions: List[Any] = []
        species_at_sites: List[str] = []
        for site in sites:
            xyz = site.get("xyz")
            if xyz is None:
                return None
            cartesian_site_positions.append(xyz)

            label = site.get("label")
            if isinstance(label, str) and label:
                species_at_sites.append(label)
                continue

            species = site.get("species", [])
            if not species:
                return None
            elem = species[0].get("element")
            if not elem:
                return None
            species_at_sites.append(str(elem))

        band_gap = data.get("band_gap_ind")
        if band_gap is None:
            band_gap = data.get("band_gap_dir")

        attrs = {
            "chemical_formula_reduced": data.get("formula"),
            "elements": data.get("elements"),
            "nsites": data.get("nsites"),
            "lattice_vectors": lattice_vectors,
            "cartesian_site_positions": cartesian_site_positions,
            "species_at_sites": species_at_sites,
            "last_modified": data.get("last_modified"),
            "_alexandria_hull_distance": data.get("e_above_hull"),
            "_alexandria_formation_energy_per_atom": data.get("e_form"),
            "_alexandria_energy": data.get("energy_total"),
            "_alexandria_energy_corrected": data.get("energy_corrected"),
            "_alexandria_band_gap": band_gap,
            "_alexandria_space_group": data.get("spg"),
        }
        return {"id": str(material_id), "attributes": attrs}

    def _matches_filters(self, doc: Dict[str, Any]) -> bool:
        attrs = doc.get("attributes", {})

        if self.elements:
            doc_elements = attrs.get("elements")
            if not isinstance(doc_elements, list):
                return False
            if not self.elements.issubset(set(str(e) for e in doc_elements)):
                return False

        if self.stable_only:
            e_hull = attrs.get("_alexandria_hull_distance")
            if e_hull is None:
                return False
            try:
                e_hull = float(e_hull)
            except (TypeError, ValueError):
                return False
            if e_hull > self.stable_threshold:
                return False

        return True

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        if not self.local_dir.exists() or not self.local_dir.is_dir():
            raise FileNotFoundError(
                f"Alexandria local directory not found: {self.local_dir}"
            )

        chunk_files = sorted(self.local_dir.glob(self.file_pattern))
        if not chunk_files:
            raise FileNotFoundError(
                f"No files matching '{self.file_pattern}' in {self.local_dir}"
            )

        logger.info("Reading Alexandria data from local chunk files...")
        logger.info(f"  Folder: {self.local_dir}")
        logger.info(
            f"  Found {len(chunk_files)} chunks matching pattern: {self.file_pattern}"
        )
        if self.elements:
            logger.info(f"  Filtering to elements: {', '.join(sorted(self.elements))}")
        if self.stable_only:
            logger.info(
                f"  Filtering to stable only (hull distance <= {self.stable_threshold})"
            )

        total_entries_seen = 0
        for chunk_file in chunk_files:
            logger.info(f"  Loading {chunk_file.name}")
            try:
                with bz2.open(chunk_file, "rt") as f:
                    payload = json.load(f)
            except Exception as e:
                logger.warning(f"  Failed to read {chunk_file.name}: {e}")
                continue

            entries = payload.get("entries", [])
            total_entries_seen += len(entries)
            self.total_available = total_entries_seen

            for entry in entries:
                doc = self._to_doc(entry)
                if doc is None:
                    continue
                if not self._matches_filters(doc):
                    continue
                self.fetched_count += 1
                yield doc


def import_mp_material(
    db: StructureDatabase,
    doc: Any,
    defer_commit: bool = False,
) -> bool:
    """
    Import a single MP material into the database.

    Returns True if imported, False if skipped.
    """
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    material_id = str(doc.material_id)
    structure = doc.structure

    if structure is None:
        logger.debug(f"  Skipping {material_id}: no structure")
        return False

    # Get formula and composition
    formula = doc.formula_pretty or structure.composition.reduced_formula
    composition = structure.composition
    stoichiometry = {str(el): int(amt) for el, amt in composition.element_composition.items()}
    num_atoms = structure.num_sites

    # Get space group info
    sg_number = None
    sg_symbol = None
    if doc.symmetry:
        sg_number = getattr(doc.symmetry, "number", None)
        sg_symbol = getattr(doc.symmetry, "symbol", None)

    # Fall back to analyzer if symmetry not in summary
    if sg_number is None:
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number = analyzer.get_space_group_number()
            sg_symbol = analyzer.get_space_group_symbol()
        except Exception:
            pass

    # Generate CIF content
    try:
        cif_writer = CifWriter(structure, symprec=0.1)
        cif_content = str(cif_writer)
    except Exception:
        try:
            cif_content = structure.to(fmt="cif")
        except Exception as e:
            logger.debug(f"  Skipping {material_id}: CIF generation failed: {e}")
            return False

    # Store original MP data in metadata
    mp_energy_per_atom = doc.energy_per_atom if hasattr(doc, "energy_per_atom") else None
    mp_energy_above_hull = doc.energy_above_hull if hasattr(doc, "energy_above_hull") else None
    mp_is_stable = doc.is_stable if hasattr(doc, "is_stable") else None

    generation_metadata = {
        "mp_material_id": material_id,
        "mp_energy_per_atom": mp_energy_per_atom,
        "mp_energy_above_hull": mp_energy_above_hull,
        "mp_is_stable": mp_is_stable,
    }

    # Add to database with energy_per_atom=None (will be set by ORB relaxation)
    db.add_structure(
        formula=formula,
        stoichiometry=stoichiometry,
        energy_per_atom=None,
        total_energy=None,
        num_atoms=num_atoms,
        space_group_number=sg_number,
        space_group_symbol=sg_symbol,
        cif_content=cif_content,
        is_valid=True,
        generation_metadata=generation_metadata,
        source="mp",
        source_id=build_source_id("mp", material_id),
        defer_commit=defer_commit,
    )

    return True


def import_alexandria_material(
    db: StructureDatabase,
    doc: Dict[str, Any],
    defer_commit: bool = False,
) -> bool:
    """
    Import a single Alexandria OPTIMADE structure into the database.

    Returns True if imported, False if skipped.
    """
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    material_id = str(doc.get("id", ""))
    attrs = doc.get("attributes", {})

    lattice_vectors = attrs.get("lattice_vectors")
    cartesian_site_positions = attrs.get("cartesian_site_positions")
    species_at_sites = attrs.get("species_at_sites")

    if not material_id or lattice_vectors is None or cartesian_site_positions is None:
        logger.debug("  Skipping entry: missing required OPTIMADE structure fields")
        return False
    if not species_at_sites:
        logger.debug(f"  Skipping {material_id}: missing species_at_sites")
        return False

    try:
        structure = Structure(
            lattice=lattice_vectors,
            species=species_at_sites,
            coords=cartesian_site_positions,
            coords_are_cartesian=True,
        )
    except Exception as e:
        logger.debug(f"  Skipping {material_id}: invalid structure payload: {e}")
        return False

    formula = attrs.get("chemical_formula_reduced") or structure.composition.reduced_formula
    composition = structure.composition
    stoichiometry = {str(el): int(amt) for el, amt in composition.element_composition.items()}
    num_atoms = structure.num_sites

    sg_number = None
    sg_symbol = None
    raw_sg = attrs.get("_alexandria_space_group")
    if isinstance(raw_sg, dict):
        sg_number = raw_sg.get("number")
        sg_symbol = raw_sg.get("symbol")
    elif isinstance(raw_sg, int):
        sg_number = raw_sg
    elif isinstance(raw_sg, str):
        sg_symbol = raw_sg

    if sg_number is None:
        try:
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number = analyzer.get_space_group_number()
            if sg_symbol is None:
                sg_symbol = analyzer.get_space_group_symbol()
        except Exception:
            pass

    try:
        cif_writer = CifWriter(structure, symprec=0.1)
        cif_content = str(cif_writer)
    except Exception:
        try:
            cif_content = structure.to(fmt="cif")
        except Exception as e:
            logger.debug(f"  Skipping {material_id}: CIF generation failed: {e}")
            return False

    generation_metadata = {
        "alexandria_material_id": material_id,
        "alexandria_hull_distance": attrs.get("_alexandria_hull_distance"),
        "alexandria_formation_energy_per_atom": attrs.get(
            "_alexandria_formation_energy_per_atom"
        ),
        "alexandria_energy": attrs.get("_alexandria_energy"),
        "alexandria_energy_corrected": attrs.get("_alexandria_energy_corrected"),
        "alexandria_band_gap": attrs.get("_alexandria_band_gap"),
        "alexandria_last_modified": attrs.get("last_modified"),
    }

    db.add_structure(
        formula=formula,
        stoichiometry=stoichiometry,
        energy_per_atom=None,
        total_energy=None,
        num_atoms=num_atoms,
        space_group_number=sg_number,
        space_group_symbol=sg_symbol,
        cif_content=cif_content,
        is_valid=True,
        generation_metadata=generation_metadata,
        source="alexandria",
        source_id=build_source_id("alexandria", material_id),
        defer_commit=defer_commit,
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Import MP or Alexandria structures into ggen database"
    )
    parser.add_argument(
        "--database",
        type=str,
        default="ggen.db",
        help="Path to ggen database file (default: ggen.db)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["mp", "alexandria"],
        default="mp",
        help="Dataset provider to import from (default: mp)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Materials Project API key (only used for provider=mp)",
    )
    parser.add_argument(
        "--alexandria-url",
        type=str,
        default="https://alexandria.icams.rub.de/pbe/v1/structures",
        help="Alexandria OPTIMADE structures endpoint URL",
    )
    parser.add_argument(
        "--alexandria-local-dir",
        type=str,
        default=None,
        help=(
            "Read Alexandria from local .json.bz2 chunks in this folder "
            "(bypasses API)"
        ),
    )
    parser.add_argument(
        "--alexandria-local-pattern",
        type=str,
        default="alexandria_*.json.bz2",
        help="Glob pattern for local Alexandria chunk files (default: alexandria_*.json.bz2)",
    )
    parser.add_argument(
        "--alexandria-stable-threshold",
        type=float,
        default=1e-8,
        help="Stable threshold for Alexandria hull distance (default: 1e-8)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=None,
        help="Filter to materials containing these elements (e.g., --elements Fe Mn Co)",
    )
    parser.add_argument(
        "--stable-only",
        action="store_true",
        help="Only import thermodynamically stable materials (on the MP hull)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="API fetch chunk size (default: 1000; Alexandria capped to 500)",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=500,
        help="Commit to database every N structures (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without actually importing",
    )
    args = parser.parse_args()

    C = Colors

    source = args.provider

    # Resolve API key (MP only)
    api_key: Optional[str] = None
    if source == "mp":
        api_key = args.api_key or os.getenv("MP_API_KEY")
        if not api_key:
            logger.error(
                f"{C.RED}No API key provided. Set MP_API_KEY env var or use --api-key.{C.RESET}"
            )
            sys.exit(1)

    source_label = "Materials Project" if source == "mp" else "Alexandria"
    logger.info(f"{C.BOLD}{source_label} Import{C.RESET}")
    logger.info(f"{C.DIM}{'=' * 50}{C.RESET}")
    logger.info(f"Database: {C.CYAN}{args.database}{C.RESET}")

    # Open database
    try:
        db = StructureDatabase(args.database)
    except Exception as e:
        logger.error(f"{C.RED}Failed to open database: {e}{C.RESET}")
        sys.exit(1)

    # Count already-imported materials without loading all IDs into memory
    already_imported_count = count_already_imported(db, source)
    if already_imported_count:
        logger.info(
            f"Already imported: {C.YELLOW}{already_imported_count}{C.RESET} {source} structures"
        )

    # Build the doc iterator and known total for each provider.
    # MP loads all docs up-front (manageable size); Alexandria streams page-by-page.
    logger.info("")
    alex_stream: Optional[Any] = None
    if source == "mp":
        docs: List[Any] = fetch_mp_summaries(
            api_key=api_key or "",
            elements=args.elements,
            stable_only=args.stable_only,
            chunk_size=args.chunk_size,
        )
        doc_iter = iter(docs)
        total_known: Optional[int] = len(docs)
    else:
        if args.alexandria_local_dir:
            alex_stream = LocalAlexandriaStream(
                local_dir=args.alexandria_local_dir,
                elements=args.elements,
                stable_only=args.stable_only,
                stable_threshold=args.alexandria_stable_threshold,
                file_pattern=args.alexandria_local_pattern,
            )
        else:
            alex_stream = AlexandriaStream(
                endpoint_url=args.alexandria_url,
                elements=args.elements,
                stable_only=args.stable_only,
                chunk_size=args.chunk_size,
                stable_threshold=args.alexandria_stable_threshold,
                timeout=args.request_timeout,
            )
        doc_iter = iter(alex_stream)
        total_known = None  # set from first OPTIMADE response
        docs = []  # unused; keeps del docs below safe

    def _get_material_id(doc: Any) -> str:
        if source == "mp":
            return str(doc.material_id)
        return str(doc.get("id", ""))

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    if args.dry_run:
        logger.info(f"{C.BOLD}Dry run -- would import:{C.RESET}")
        sample_docs: List[Any] = []
        skipped_existing = 0
        new_count = 0
        total_seen = 0
        existing_source_ids = load_existing_source_ids(db, source)
        logger.info(
            f"Loaded {len(existing_source_ids)} existing {source} source IDs for dedupe"
        )

        dry_desc = f"Dry-run scanning {source} structures"
        dry_pbar = tqdm(doc_iter, total=total_known, desc=dry_desc)
        for doc in dry_pbar:
            if (
                alex_stream is not None
                and dry_pbar.total is None
                and alex_stream.total_available is not None
            ):
                dry_pbar.total = alex_stream.total_available
                dry_pbar.refresh()

            total_seen += 1
            material_id = _get_material_id(doc)
            source_id = build_source_id(source, material_id)
            if source_id in existing_source_ids:
                skipped_existing += 1
            else:
                new_count += 1
                if len(sample_docs) < 20:
                    sample_docs.append(doc)
            if total_seen % 500 == 0:
                gc.collect()
                dry_pbar.set_postfix(
                    new=new_count,
                    existing=skipped_existing,
                    refresh=False,
                )
        dry_pbar.close()

        logger.info(f"New materials to import: {C.GREEN}{new_count}{C.RESET}")
        if skipped_existing > 0:
            logger.info(
                f"Skipping {C.DIM}{skipped_existing}{C.RESET} already-imported materials"
            )
        logger.info("")

        for doc in sample_docs:
            if source == "mp":
                formula = doc.formula_pretty or "?"
                nsites = doc.nsites if hasattr(doc, "nsites") else "?"
                e_hull = (
                    doc.energy_above_hull if hasattr(doc, "energy_above_hull") else None
                )
                e_hull_str = f"E_hull={e_hull * 1000:.0f} meV" if e_hull is not None else ""
                material_id = str(doc.material_id)
            else:
                attrs = doc.get("attributes", {})
                formula = attrs.get("chemical_formula_reduced", "?")
                nsites = attrs.get("nsites", "?")
                e_hull = attrs.get("_alexandria_hull_distance")
                e_hull_str = f"E_hull={e_hull * 1000:.0f} meV" if e_hull is not None else ""
                material_id = str(doc.get("id", ""))
            logger.info(f"  {material_id:12s}  {formula:16s}  sites={nsites}  {e_hull_str}")
        if new_count > len(sample_docs):
            logger.info(f"  ... and {new_count - len(sample_docs)} more")

        del sample_docs, docs
        db.close()
        return

    # ------------------------------------------------------------------
    # Import structures
    # ------------------------------------------------------------------
    logger.info(f"{C.BOLD}Importing structures...{C.RESET}")
    logger.info(f"{C.DIM}{'-' * 50}{C.RESET}")

    skipped_existing = 0
    new_count = 0
    num_imported = 0
    num_failed = 0
    uncommitted = 0
    commit_every = args.commit_every

    progress_desc = f"Importing {source} structures"
    pbar = tqdm(doc_iter, total=total_known, desc=progress_desc)

    for doc in pbar:
        # Update total from OPTIMADE meta on first page (Alexandria only)
        if (
            alex_stream is not None
            and pbar.total is None
            and alex_stream.total_available is not None
        ):
            pbar.total = alex_stream.total_available
            pbar.refresh()

        material_id = _get_material_id(doc)
        if is_already_imported(db, source, material_id):
            skipped_existing += 1
            continue

        new_count += 1
        try:
            if source == "mp":
                success = import_mp_material(db, doc, defer_commit=True)
            else:
                success = import_alexandria_material(db, doc, defer_commit=True)
            if success:
                num_imported += 1
                uncommitted += 1
            else:
                num_failed += 1
        except Exception as e:
            logger.debug(f"  Failed {material_id}: {e}")
            num_failed += 1

        # Periodic commit
        if uncommitted >= commit_every:
            db.conn.commit()
            uncommitted = 0

    pbar.close()

    # Final commit for any remaining rows
    if uncommitted > 0:
        db.conn.commit()

    del docs
    total_fetched = alex_stream.fetched_count if alex_stream is not None else total_known

    # Final summary
    logger.info("")
    logger.info(f"{C.BOLD}Import Summary{C.RESET}")
    logger.info(f"{C.DIM}{'-' * 50}{C.RESET}")
    logger.info(f"  Total fetched:    {total_fetched}")
    logger.info(f"  Already existed:  {skipped_existing}")
    logger.info(f"  New candidates:   {new_count}")
    logger.info(f"  Newly imported:   {C.GREEN}{num_imported}{C.RESET}")
    if num_failed:
        logger.info(f"  Failed:           {C.YELLOW}{num_failed}{C.RESET}")
    logger.info("")

    # Show next steps
    total_source = already_imported_count + num_imported
    logger.info(f"{C.BOLD}Total {source} structures in database: {total_source}{C.RESET}")
    logger.info("")
    logger.info(f"{C.CYAN}Next step:{C.RESET} Relax imported structures with ORB:")
    logger.info(
        f"  python scripts/relax.py --source {source} --unrelaxed --batch-size 32"
    )

    db.close()
    logger.info(f"{C.GREEN}{C.BOLD}Done!{C.RESET}")


if __name__ == "__main__":
    main()
