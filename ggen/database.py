"""
Structure database shared by GGen and the materials services.

One SQLite file holds every structure Ouro knows about: reference data
(Materials Project, Alexandria), GGen output and user submissions. Each
structure can carry energies on several energy surfaces (see
``ggen.surfaces``): PBE from the source databases and any number of MLIPs.
A ``StructureDatabase`` reads and writes one surface at a time, chosen when
it is opened, so callers see the same ``energy_per_atom`` / ``e_above_hull``
fields whichever surface they work on.

Tables (schema version 2):

- ``structures``: identity only (source, formula, symmetry, metadata).
- ``geometries``: zlib-compressed CIF per (structure, kind), where kind is
  ``source`` (as published) or a surface id (relaxed on that surface).
- ``calculations``: per (structure, surface) raw energy, MP2020 correction,
  hull distance, and phonon results.
- ``runs`` / ``run_structures``: GGen exploration runs.

Hulls are built from corrected energies (raw + MP2020) of one surface, from
the lowest-energy structure of each reduced formula. ``e_above_hull`` is
stored for every structure with an energy, polymorphs included.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from itertools import combinations
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry

from .surfaces import DEFAULT_SURFACE, SOURCE_GEOMETRY, get_surface

if TYPE_CHECKING:
    from .matching import Candidate, NoveltyIndex

logger = logging.getLogger(__name__)

T = TypeVar("T")

SCHEMA_VERSION = 2
ON_HULL_TOLERANCE = 1e-6


def _retry_on_lock(
    max_retries: int = 5,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry database operations on lock errors.

    Uses exponential backoff with jitter to handle concurrent access.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            delay = base_delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() or "busy" in str(e).lower():
                        last_error = e
                        if attempt < max_retries - 1:
                            # Add jitter to prevent thundering herd
                            jitter = delay * 0.2 * (0.5 - time.time() % 1)
                            sleep_time = min(delay + jitter, max_delay)
                            logger.debug(
                                f"Database locked, retrying in {sleep_time:.2f}s "
                                f"(attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(sleep_time)
                            delay = min(delay * 2, max_delay)
                    else:
                        raise

            raise sqlite3.OperationalError(
                f"Database still locked after {max_retries} retries: {last_error}"
            )

        return wrapper

    return decorator


def compress_cif(cif: str) -> bytes:
    return zlib.compress(cif.encode(), 6)


def decompress_cif(blob: Optional[bytes]) -> Optional[str]:
    if blob is None:
        return None
    return zlib.decompress(blob).decode()


def _now() -> str:
    return datetime.now().isoformat()


def _bool_or_none(value: Optional[int]) -> Optional[bool]:
    return None if value is None else bool(value)


@dataclass
class StoredStructure:
    """A structure stored in the database, with its values on one surface."""

    id: str
    formula: str
    elements: List[str]
    chemsys: str
    stoichiometry: Dict[str, int]

    # Raw energies on ``surface`` (None if not computed there).
    energy_per_atom: Optional[float]
    total_energy: Optional[float]
    num_atoms: int

    space_group_number: Optional[int] = None
    space_group_symbol: Optional[str] = None

    structure_hash: Optional[str] = None
    cif_content: Optional[str] = None

    is_valid: bool = True
    error_message: Optional[str] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

    # Provenance tracking
    source: str = "ggen"  # 'ggen', 'mp', 'alexandria', 'ouro', 'matra-genoa'
    source_id: Optional[str] = None  # Original identifier, e.g. 'mp-1234'

    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Hull info on ``surface``
    e_above_hull: Optional[float] = None
    is_on_hull: bool = False

    # Dynamical stability (phonon) info on ``surface``
    is_dynamically_stable: Optional[bool] = None
    num_imaginary_modes: Optional[int] = None
    min_phonon_frequency: Optional[float] = None
    max_phonon_frequency: Optional[float] = None
    phonon_supercell: Optional[str] = None  # JSON string of supercell tuple

    reduced_formula: Optional[str] = None
    surface: Optional[str] = None
    # MP2020 adjustment per atom; None if not computed or MP2020 rejects it.
    correction_per_atom: Optional[float] = None

    # Loaded structure object (not stored in DB)
    structure: Optional[Structure] = None

    def __post_init__(self) -> None:
        if self.reduced_formula is None:
            self.reduced_formula = Composition(self.formula).reduced_formula

    @property
    def corrected_energy_per_atom(self) -> Optional[float]:
        if self.energy_per_atom is None or self.correction_per_atom is None:
            return None
        return self.energy_per_atom + self.correction_per_atom

    def get_structure(self) -> Optional[Structure]:
        """Get the pymatgen Structure, loading from CIF if needed."""
        if self.structure is not None:
            return self.structure
        if self.cif_content:
            try:
                self.structure = Structure.from_str(self.cif_content, fmt="cif")
                return self.structure
            except Exception as e:
                logger.warning(f"Failed to parse CIF for {self.formula}: {e}")
        return None


@dataclass
class ExplorationRun:
    """Record of an exploration run."""

    id: str
    chemical_system: str
    elements: List[str]
    parameters: Dict[str, Any]

    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    num_candidates: int = 0
    num_successful: int = 0
    num_failed: int = 0

    notes: Optional[str] = None


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS structures (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL DEFAULT 'ggen',
        source_id TEXT,
        formula TEXT NOT NULL,
        reduced_formula TEXT NOT NULL,
        elements TEXT NOT NULL,
        chemsys TEXT NOT NULL,
        stoichiometry TEXT NOT NULL,
        num_atoms INTEGER NOT NULL,
        space_group_number INTEGER,
        space_group_symbol TEXT,
        structure_hash TEXT,
        is_valid INTEGER NOT NULL DEFAULT 1,
        error_message TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS geometries (
        structure_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        cif BLOB NOT NULL,
        PRIMARY KEY (structure_id, kind)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS calculations (
        structure_id TEXT NOT NULL,
        surface TEXT NOT NULL,
        energy_per_atom REAL,
        total_energy REAL,
        correction_per_atom REAL,
        run_type TEXT,
        hubbards TEXT,
        e_above_hull REAL,
        is_on_hull INTEGER,
        hull_computed_at TEXT,
        is_dynamically_stable INTEGER,
        num_imaginary_modes INTEGER,
        min_phonon_frequency REAL,
        max_phonon_frequency REAL,
        phonon_supercell TEXT,
        metadata TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (structure_id, surface)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        chemical_system TEXT NOT NULL,
        elements TEXT NOT NULL,
        surface TEXT,
        parameters TEXT,
        started_at TEXT,
        completed_at TEXT,
        num_candidates INTEGER DEFAULT 0,
        num_successful INTEGER DEFAULT 0,
        num_failed INTEGER DEFAULT 0,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS run_structures (
        run_id TEXT NOT NULL,
        structure_id TEXT NOT NULL,
        e_above_hull REAL,
        is_on_hull INTEGER DEFAULT 0,
        PRIMARY KEY (run_id, structure_id)
    ) WITHOUT ROWID
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_structures_source_id "
    "ON structures(source, source_id) WHERE source_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_structures_chemsys ON structures(chemsys)",
    "CREATE INDEX IF NOT EXISTS idx_structures_reduced_formula "
    "ON structures(reduced_formula)",
    "CREATE INDEX IF NOT EXISTS idx_structures_formula ON structures(formula)",
    "CREATE INDEX IF NOT EXISTS idx_structures_hash ON structures(structure_hash)",
    "CREATE INDEX IF NOT EXISTS idx_structures_updated ON structures(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_calculations_hull "
    "ON calculations(surface, e_above_hull)",
    "CREATE INDEX IF NOT EXISTS idx_calculations_updated ON calculations(updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_run_structures_structure "
    "ON run_structures(structure_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_system ON runs(chemical_system)",
]

# Columns read into a StoredStructure. ``s`` is structures, ``c`` is the
# calculation on the database's surface (LEFT JOIN, so it may be NULL).
_STRUCTURE_COLUMNS = """
    s.id, s.source, s.source_id, s.formula, s.reduced_formula, s.elements,
    s.chemsys, s.stoichiometry, s.num_atoms, s.space_group_number,
    s.space_group_symbol, s.structure_hash, s.is_valid, s.error_message,
    s.metadata, s.created_at, s.updated_at,
    c.energy_per_atom, c.total_energy, c.correction_per_atom, c.e_above_hull,
    c.is_on_hull, c.is_dynamically_stable, c.num_imaginary_modes,
    c.min_phonon_frequency, c.max_phonon_frequency, c.phonon_supercell
"""

# The geometry shown for a structure on a surface: relaxed on that surface if
# available, else as published.
_GEOMETRY_COLUMN = """
    COALESCE(
        (SELECT g.cif FROM geometries g WHERE g.structure_id = s.id AND g.kind = :surface),
        (SELECT g.cif FROM geometries g WHERE g.structure_id = s.id AND g.kind = 'source')
    ) AS cif
"""


class StructureDatabase:
    """
    Database of crystal structures and their energies on one energy surface.

    Example:
        >>> db = StructureDatabase("./ggen.db")
        >>> db.add_structure(formula="Fe3Mn", energy_per_atom=-8.5, ...)
        >>> best = db.get_best_structures_for_subsystem("Fe-Mn")
        >>> db.compute_hull("Fe-Mn-Co")  # Uses all relevant structures

    Args:
        db_path: SQLite file. Defaults to ./ggen.db.
        surface: Energy surface energies are read from and written to.
        readonly: Open without write access (and without creating the schema).
        snapshot: With ``readonly``, also promise the file will not change
            while open (``immutable=1``), which skips locking entirely. Use
            for a Modal Volume view, which only changes on ``reload()``.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        surface: str = DEFAULT_SURFACE,
        readonly: bool = False,
        snapshot: bool = False,
    ):
        if db_path is None:
            db_path = Path.cwd() / "ggen.db"
        else:
            db_path = Path(db_path).expanduser()

        self.surface = get_surface(surface).id
        self.readonly = readonly
        self.snapshot = snapshot and readonly
        if readonly and not db_path.exists():
            raise FileNotFoundError(f"No database at {db_path}")
        if not readonly:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path

        self._conn: Optional[sqlite3.Connection] = None
        if readonly:
            self._check_schema()
        else:
            self._init_database()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection, creating if needed.

        Writers use WAL mode and a busy timeout so concurrent readers in the
        same container never block. Readers open read-only and query-only.
        """
        if self._conn is None:
            if self.readonly:
                flags = "mode=ro&immutable=1" if self.snapshot else "mode=ro"
                self._conn = sqlite3.connect(
                    f"file:{self.db_path}?{flags}", uri=True, timeout=60.0
                )
                self._conn.execute("PRAGMA query_only=ON")
            else:
                is_new = not self.db_path.exists() or self.db_path.stat().st_size == 0
                self._conn = sqlite3.connect(
                    str(self.db_path), timeout=60.0, isolation_level="DEFERRED"
                )
                if is_new:
                    # Larger pages suit this read-mostly, multi-GB file; the
                    # page size can only be set before the first table exists.
                    self._conn.execute("PRAGMA page_size=8192")
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA busy_timeout=60000")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA cache_size=-131072")  # 128 MiB
            self._conn.execute("PRAGMA mmap_size=1073741824")
            self._conn.execute("PRAGMA temp_store=MEMORY")
        return self._conn

    def _table_columns(self, table: str, schema: str = "main") -> List[str]:
        return [
            row[1] for row in self.conn.execute(f"PRAGMA {schema}.table_info({table})")
        ]

    def _check_schema(self) -> None:
        columns = self._table_columns("structures")
        if "cif_content" in columns:
            raise RuntimeError(
                f"{self.db_path} uses the version 1 schema; migrate it with "
                "scripts/migrate_v2.py"
            )

    def _init_database(self) -> None:
        """Create the schema, refusing to touch a version 1 database."""
        self._check_schema()
        conn = self.conn
        for statement in _SCHEMA:
            conn.execute(statement)
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        logger.info(f"Initialized database at {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            if not self.readonly:
                try:
                    self._conn.execute("PRAGMA optimize")
                except sqlite3.Error:
                    pass
            self._conn.close()
            self._conn = None

    @_retry_on_lock(max_retries=5, base_delay=0.1, max_delay=2.0)
    def _commit(self) -> None:
        """Commit with retry on lock errors."""
        self.conn.commit()

    def analyze(self) -> None:
        """Refresh the query planner's statistics (after bulk loads)."""
        self.conn.execute("ANALYZE")
        self._commit()

    # -------------------- Subsystem Utilities --------------------

    @staticmethod
    def get_all_subsystems(elements: List[str]) -> List[str]:
        """
        Get all subsystems for a set of elements.

        For [Co, Fe, Mn], returns:
        ["Co", "Fe", "Mn", "Co-Fe", "Co-Mn", "Fe-Mn", "Co-Fe-Mn"]
        """
        elements = sorted(elements)
        subsystems = []
        for r in range(1, len(elements) + 1):
            for combo in combinations(elements, r):
                subsystems.append("-".join(combo))
        return subsystems

    @staticmethod
    def normalize_chemsys(system: str) -> str:
        """Normalize a chemical system string (alphabetically sorted)."""
        from .utils import normalize_dashes

        elements = [
            e.strip() for e in normalize_dashes(system).replace("-", " ").split()
        ]
        return "-".join(sorted(elements))

    @staticmethod
    def compute_structure_hash(
        formula: str,
        energy_per_atom: float,
        space_group: Optional[int] = None,
        source: str = "ggen",
        source_id: Optional[str] = None,
    ) -> str:
        """Compute a hash for structure deduplication.

        Includes source so that structures from different origins
        (e.g. ggen-generated vs Materials Project) are kept separate.
        When source_id is provided (external imports), it is included
        so that distinct structures sharing formula/SG/energy don't collide.
        """
        key = f"{source}:{formula}:{energy_per_atom:.6f}:{space_group or 'unknown'}"
        if source_id:
            key = f"{key}:{source_id}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _subsystem_clause(self, chemical_system: str) -> Tuple[str, List[str]]:
        chemsys = self.normalize_chemsys(chemical_system)
        subsystems = self.get_all_subsystems(chemsys.split("-"))
        return f"s.chemsys IN ({','.join('?' * len(subsystems))})", subsystems

    # -------------------- Structure Operations --------------------

    def add_structure(
        self,
        formula: str,
        stoichiometry: Dict[str, int],
        energy_per_atom: Optional[float],
        total_energy: Optional[float],
        num_atoms: int,
        space_group_number: Optional[int] = None,
        space_group_symbol: Optional[str] = None,
        cif_content: Optional[str] = None,
        structure: Optional[Structure] = None,
        is_valid: bool = True,
        error_message: Optional[str] = None,
        generation_metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        # Provenance
        source: str = "ggen",
        source_id: Optional[str] = None,
        # Phonon stability fields
        is_dynamically_stable: Optional[bool] = None,
        num_imaginary_modes: Optional[int] = None,
        min_phonon_frequency: Optional[float] = None,
        max_phonon_frequency: Optional[float] = None,
        phonon_supercell: Optional[Tuple[int, int, int]] = None,
        # Where the geometry belongs; defaults to this surface when an energy
        # is given (the structure was relaxed here), else the source slot.
        geometry_kind: Optional[str] = None,
        # Commit control
        check_existing: bool = True,
        defer_commit: bool = False,
    ) -> str:
        """
        Add a structure, with its energy on this database's surface if given.

        Args:
            formula: Chemical formula (e.g., "Fe3Mn")
            stoichiometry: Element counts {"Fe": 3, "Mn": 1}
            energy_per_atom: Raw energy per atom in eV on this surface (None
                for structures imported without an energy here)
            total_energy: Total energy in eV
            num_atoms: Number of atoms
            space_group_number: International space group number
            space_group_symbol: Space group symbol
            cif_content: CIF file content as string
            structure: pymatgen Structure object (used to generate CIF if not provided)
            is_valid: Whether the structure is valid
            error_message: Error message if invalid
            generation_metadata: Additional metadata dict
            run_id: Optional run ID to link this structure to
            source: Data source ('ggen', 'mp', 'alexandria', 'ouro', 'matra-genoa')
            source_id: Original identifier from the source (e.g., 'mp-1234')
            is_dynamically_stable: Whether structure is dynamically stable (no imaginary phonon modes)
            num_imaginary_modes: Number of imaginary phonon modes
            min_phonon_frequency: Minimum phonon frequency in THz
            max_phonon_frequency: Maximum phonon frequency in THz
            phonon_supercell: Supercell dimensions used for phonon calculation
            geometry_kind: Geometry slot for the CIF (see above).
            check_existing: If True, run dedupe checks (source_id/hash) before insert.
                Set to False only when caller already guarantees uniqueness.
            defer_commit: If True, skip the automatic commit after insertion.
                Caller is responsible for calling db.conn.commit() periodically.

        Returns:
            Structure ID (UUID)
        """
        elements = sorted(stoichiometry.keys())
        chemsys = "-".join(elements)

        if cif_content is None and structure is not None:
            try:
                cif_content = structure.to(fmt="cif")
            except Exception as e:
                logger.warning(f"Failed to generate CIF for {formula}: {e}")

        if check_existing and source_id:
            existing_by_source = self.conn.execute(
                "SELECT id FROM structures WHERE source = ? AND source_id = ?",
                (source, source_id),
            ).fetchone()
            if existing_by_source:
                logger.debug(f"Structure {source_id} already imported, skipping")
                return existing_by_source["id"]

        structure_hash = self.compute_structure_hash(
            formula, energy_per_atom or 0.0, space_group_number, source, source_id
        )

        if check_existing:
            existing = self.conn.execute(
                """
                SELECT s.id, c.energy_per_atom FROM structures s
                LEFT JOIN calculations c ON c.structure_id = s.id AND c.surface = ?
                WHERE s.structure_hash = ?
                """,
                (self.surface, structure_hash),
            ).fetchone()

            if existing:
                existing_energy = existing["energy_per_atom"]
                if (
                    energy_per_atom is not None
                    and existing_energy is not None
                    and energy_per_atom < existing_energy
                ):
                    self._update_structure(
                        existing["id"],
                        energy_per_atom=energy_per_atom,
                        total_energy=total_energy,
                        cif_content=cif_content,
                        generation_metadata=generation_metadata,
                        is_dynamically_stable=is_dynamically_stable,
                        num_imaginary_modes=num_imaginary_modes,
                        min_phonon_frequency=min_phonon_frequency,
                        max_phonon_frequency=max_phonon_frequency,
                        phonon_supercell=phonon_supercell,
                    )
                    logger.debug(f"Updated existing structure {formula} with lower energy")
                return existing["id"]

        structure_id = str(uuid.uuid4())
        now = _now()
        self.conn.execute(
            """
            INSERT INTO structures (
                id, source, source_id, formula, reduced_formula, elements, chemsys,
                stoichiometry, num_atoms, space_group_number, space_group_symbol,
                structure_hash, is_valid, error_message, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                structure_id,
                source,
                source_id,
                formula,
                Composition(stoichiometry).reduced_formula,
                json.dumps(elements),
                chemsys,
                json.dumps(stoichiometry),
                num_atoms,
                space_group_number,
                space_group_symbol,
                structure_hash,
                1 if is_valid else 0,
                error_message,
                json.dumps(generation_metadata) if generation_metadata else None,
                now,
                now,
            ),
        )

        if cif_content:
            kind = geometry_kind or (
                self.surface if energy_per_atom is not None else SOURCE_GEOMETRY
            )
            self.put_geometry(structure_id, kind, cif_content, defer_commit=True)

        has_phonons = any(
            v is not None
            for v in (
                is_dynamically_stable,
                num_imaginary_modes,
                min_phonon_frequency,
                max_phonon_frequency,
                phonon_supercell,
            )
        )
        if energy_per_atom is not None or has_phonons:
            self.set_calculation(
                structure_id,
                energy_per_atom=energy_per_atom,
                total_energy=total_energy,
                structure=structure,
                cif_content=cif_content,
                defer_commit=True,
            )
            if has_phonons:
                self._set_phonons(
                    structure_id,
                    self.surface,
                    is_dynamically_stable=is_dynamically_stable,
                    num_imaginary_modes=num_imaginary_modes,
                    min_phonon_frequency=min_phonon_frequency,
                    max_phonon_frequency=max_phonon_frequency,
                    phonon_supercell=phonon_supercell,
                )

        if not defer_commit:
            self._commit()

        if run_id:
            self.link_structure_to_run(run_id, structure_id)

        logger.debug(f"Added structure {formula} (id={structure_id[:8]}...)")
        return structure_id

    def put_geometry(
        self,
        structure_id: str,
        kind: str,
        cif_content: str,
        defer_commit: bool = False,
    ) -> None:
        """Store (or replace) a structure's geometry in one slot."""
        self.conn.execute(
            "INSERT OR REPLACE INTO geometries (structure_id, kind, cif) VALUES (?, ?, ?)",
            (structure_id, kind, compress_cif(cif_content)),
        )
        if not defer_commit:
            self._commit()

    def get_geometry(self, structure_id: str, kind: Optional[str] = None) -> Optional[str]:
        """A structure's CIF in one slot, or in any slot when ``kind`` is None."""
        if kind is None:
            row = self.conn.execute(
                "SELECT cif FROM geometries WHERE structure_id = ? ORDER BY kind LIMIT 1",
                (structure_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT cif FROM geometries WHERE structure_id = ? AND kind = ?",
                (structure_id, kind),
            ).fetchone()
        return decompress_cif(row["cif"]) if row else None

    def set_calculation(
        self,
        structure_id: str,
        energy_per_atom: Optional[float],
        *,
        surface: Optional[str] = None,
        total_energy: Optional[float] = None,
        structure: Optional[Structure] = None,
        cif_content: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        correction_per_atom: Optional[float] = None,
        compute_correction: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        defer_commit: bool = False,
    ) -> None:
        """Record a raw energy on a surface, with its MP2020 correction.

        The correction is computed from ``structure`` (or ``cif_content``, or
        the stored geometry) unless ``correction_per_atom`` is given, which
        is how a source's own correction (Materials Project's) is kept, or
        ``compute_correction`` is False because the caller already computed
        it (None then means MP2020 rejects the entry).
        ``parameters`` are the calculation's run settings when known.
        Existing phonon and hull fields are kept; the hull distance is
        cleared because the energy changed.
        """
        from .corrections import mp2020_correction

        surface = surface or self.surface
        run_type = None
        hubbards = None
        if energy_per_atom is not None and correction_per_atom is None and compute_correction:
            if structure is None:
                # MP2020 depends on composition and, for oxide types, bonding,
                # so any stored geometry of the structure will do.
                cif = (
                    cif_content
                    or self.get_geometry(structure_id, get_surface(surface).geometry_kind)
                    or self.get_geometry(structure_id, SOURCE_GEOMETRY)
                    or self.get_geometry(structure_id)
                )
                if cif:
                    try:
                        structure = Structure.from_str(cif, fmt="cif")
                    except Exception as exc:
                        logger.warning(f"Unreadable geometry for {structure_id}: {exc}")
            if structure is not None:
                correction = mp2020_correction(structure, energy_per_atom, parameters)
                correction_per_atom = correction.per_atom
                run_type = correction.run_type
                hubbards = json.dumps(correction.hubbards)
        if run_type is None and parameters:
            run_type = parameters.get("run_type")
            hubbards = json.dumps(parameters.get("hubbards") or {})

        now = _now()
        self.conn.execute(
            """
            INSERT INTO calculations (
                structure_id, surface, energy_per_atom, total_energy,
                correction_per_atom, run_type, hubbards, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (structure_id, surface) DO UPDATE SET
                energy_per_atom = excluded.energy_per_atom,
                total_energy = excluded.total_energy,
                correction_per_atom = excluded.correction_per_atom,
                run_type = excluded.run_type,
                hubbards = excluded.hubbards,
                metadata = COALESCE(excluded.metadata, calculations.metadata),
                e_above_hull = NULL,
                is_on_hull = NULL,
                hull_computed_at = NULL,
                updated_at = excluded.updated_at
            """,
            (
                structure_id,
                surface,
                energy_per_atom,
                total_energy,
                correction_per_atom,
                run_type,
                hubbards,
                json.dumps(metadata) if metadata else None,
                now,
                now,
            ),
        )
        self._touch(structure_id, now)
        if not defer_commit:
            self._commit()

    def _touch(self, structure_id: str, now: Optional[str] = None) -> None:
        self.conn.execute(
            "UPDATE structures SET updated_at = ? WHERE id = ?",
            (now or _now(), structure_id),
        )

    def _set_phonons(
        self,
        structure_id: str,
        surface: str,
        is_dynamically_stable: Optional[bool] = None,
        num_imaginary_modes: Optional[int] = None,
        min_phonon_frequency: Optional[float] = None,
        max_phonon_frequency: Optional[float] = None,
        phonon_supercell: Optional[Tuple[int, int, int]] = None,
        clear: bool = False,
    ) -> None:
        now = _now()
        self.conn.execute(
            """
            INSERT OR IGNORE INTO calculations (structure_id, surface, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (structure_id, surface, now, now),
        )
        if clear:
            self.conn.execute(
                """
                UPDATE calculations SET is_dynamically_stable = NULL,
                    num_imaginary_modes = NULL, min_phonon_frequency = NULL,
                    max_phonon_frequency = NULL, phonon_supercell = NULL,
                    updated_at = ?
                WHERE structure_id = ? AND surface = ?
                """,
                (now, structure_id, surface),
            )
            return
        updates, values = [], []
        if is_dynamically_stable is not None:
            updates.append("is_dynamically_stable = ?")
            values.append(1 if is_dynamically_stable else 0)
        if num_imaginary_modes is not None:
            updates.append("num_imaginary_modes = ?")
            values.append(num_imaginary_modes)
        if min_phonon_frequency is not None:
            updates.append("min_phonon_frequency = ?")
            values.append(min_phonon_frequency)
        if max_phonon_frequency is not None:
            updates.append("max_phonon_frequency = ?")
            values.append(max_phonon_frequency)
        if phonon_supercell is not None:
            updates.append("phonon_supercell = ?")
            values.append(
                phonon_supercell
                if isinstance(phonon_supercell, str)
                else json.dumps(phonon_supercell)
            )
        if not updates:
            return
        updates.append("updated_at = ?")
        values.extend([now, structure_id, surface])
        self.conn.execute(
            f"UPDATE calculations SET {', '.join(updates)} "
            "WHERE structure_id = ? AND surface = ?",
            values,
        )

    def _update_structure(
        self,
        structure_id: str,
        energy_per_atom: Optional[float] = None,
        total_energy: Optional[float] = None,
        cif_content: Optional[str] = None,
        generation_metadata: Optional[Dict[str, Any]] = None,
        is_dynamically_stable: Optional[bool] = None,
        num_imaginary_modes: Optional[int] = None,
        min_phonon_frequency: Optional[float] = None,
        max_phonon_frequency: Optional[float] = None,
        phonon_supercell: Optional[Tuple[int, int, int]] = None,
        clear_phonon_data: bool = False,
    ) -> None:
        """Update a structure on this surface with better data.

        A new ``cif_content`` is the geometry relaxed on this surface, and a
        new energy refers to it, so both land in this surface's slots.

        Args:
            structure_id: ID of structure to update
            energy_per_atom: New raw energy per atom on this surface
            total_energy: New total energy value
            cif_content: New CIF content (relaxed on this surface)
            generation_metadata: New metadata dict (replaces the old one)
            is_dynamically_stable: Phonon stability (True/False)
            num_imaginary_modes: Number of imaginary phonon modes
            min_phonon_frequency: Minimum phonon frequency
            max_phonon_frequency: Maximum phonon frequency
            phonon_supercell: Supercell used for phonon calculation
            clear_phonon_data: If True, set all phonon fields to NULL
        """
        now = _now()
        if cif_content is not None:
            self.put_geometry(structure_id, self.surface, cif_content, defer_commit=True)
        if generation_metadata is not None:
            self.conn.execute(
                "UPDATE structures SET metadata = ? WHERE id = ?",
                (json.dumps(generation_metadata), structure_id),
            )
        if energy_per_atom is not None:
            self.set_calculation(
                structure_id,
                energy_per_atom=energy_per_atom,
                total_energy=total_energy,
                cif_content=cif_content,
                defer_commit=True,
            )
        if clear_phonon_data or any(
            v is not None
            for v in (
                is_dynamically_stable,
                num_imaginary_modes,
                min_phonon_frequency,
                max_phonon_frequency,
                phonon_supercell,
            )
        ):
            self._set_phonons(
                structure_id,
                self.surface,
                is_dynamically_stable=is_dynamically_stable,
                num_imaginary_modes=num_imaginary_modes,
                min_phonon_frequency=min_phonon_frequency,
                max_phonon_frequency=max_phonon_frequency,
                phonon_supercell=phonon_supercell,
                clear=clear_phonon_data,
            )
        self._touch(structure_id, now)
        self._commit()

    def update_structure_space_group(
        self,
        structure_id: str,
        space_group_number: Optional[int] = None,
        space_group_symbol: Optional[str] = None,
        defer_commit: bool = False,
    ) -> None:
        """Update space group fields for an existing structure."""
        updates = ["updated_at = ?"]
        values: List[Any] = [_now()]
        if space_group_number is not None:
            updates.append("space_group_number = ?")
            values.append(space_group_number)
        if space_group_symbol is not None:
            updates.append("space_group_symbol = ?")
            values.append(space_group_symbol)
        values.append(structure_id)
        self.conn.execute(
            f"UPDATE structures SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        if not defer_commit:
            self._commit()

    # -------------------- Reads --------------------

    def query_structures(
        self,
        where: str = "1",
        params: Union[Sequence[Any], Dict[str, Any]] = (),
        *,
        include_geometry: bool = True,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        join: str = "",
    ) -> List[StoredStructure]:
        """Structures matching a WHERE clause, with their values on this surface.

        ``where`` may reference ``s`` (structures) and ``c`` (this surface's
        calculations, NULL when absent); ``join`` adds further joins. Use
        positional ``?`` parameters, or a dict for named ones.
        """
        columns = _STRUCTURE_COLUMNS
        if include_geometry:
            columns += ", " + _GEOMETRY_COLUMN
        sql = f"""
            SELECT {columns}
            FROM structures s
            LEFT JOIN calculations c ON c.structure_id = s.id AND c.surface = :surface
            {join}
            WHERE {where}
        """
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        # sqlite3 cannot mix ? and :name, so positional params are renamed.
        if isinstance(params, dict):
            bound: Dict[str, Any] = {"surface": self.surface, **params}
        else:
            bound = {"surface": self.surface}
            for i, value in enumerate(params):
                bound[f"p{i}"] = value
            parts = sql.split("?")
            sql = "".join(
                part + (f":p{i}" if i < len(parts) - 1 else "")
                for i, part in enumerate(parts)
            )
        rows = self.conn.execute(sql, bound).fetchall()
        return [self._row_to_structure(row) for row in rows]

    def get_structure(self, structure_id: str) -> Optional[StoredStructure]:
        """Get a structure by ID."""
        rows = self.query_structures("s.id = ?", (structure_id,))
        return rows[0] if rows else None

    def get_structure_by_source(
        self, source: str, source_id: str
    ) -> Optional[StoredStructure]:
        rows = self.query_structures(
            "s.source = ? AND s.source_id = ?", (source, source_id)
        )
        return rows[0] if rows else None

    def get_structures_by_formula(
        self, formula: str, valid_only: bool = True
    ) -> List[StoredStructure]:
        """Get all structures with a given formula."""
        where = "s.formula = ?" + (" AND s.is_valid = 1" if valid_only else "")
        return self.query_structures(
            where, (formula,), order_by="c.energy_per_atom IS NULL, c.energy_per_atom"
        )

    def get_best_structure(
        self, formula: str, valid_only: bool = True
    ) -> Optional[StoredStructure]:
        """Get the lowest-energy structure for a formula."""
        structures = self.get_structures_by_formula(formula, valid_only)
        return structures[0] if structures else None

    def get_structures_for_subsystem(
        self,
        subsystem: str,
        valid_only: bool = True,
        source: Optional[str] = None,
        include_geometry: bool = True,
    ) -> List[StoredStructure]:
        """
        Get all structures that belong to a chemical subsystem.

        For "Fe-Mn", returns Fe, Mn, and all Fe-Mn structures.

        Args:
            subsystem: Chemical system string (e.g., "Fe-Mn")
            valid_only: If True, only return valid structures
            source: Optional source filter ('ggen', 'mp', etc.)
            include_geometry: Load CIFs (skip when only energies are needed)
        """
        where, params = self._subsystem_clause(subsystem)
        if valid_only:
            where += " AND s.is_valid = 1"
        if source:
            where += " AND s.source = ?"
            params.append(source)
        return self.query_structures(
            where,
            params,
            include_geometry=include_geometry,
            order_by="s.formula, c.energy_per_atom",
        )

    def get_best_structures_for_subsystem(
        self,
        subsystem: str,
        valid_only: bool = True,
        source: Optional[str] = None,
    ) -> Dict[str, StoredStructure]:
        """
        Get the best (lowest energy) structure for each formula in a subsystem.

        Args:
            subsystem: Chemical system string (e.g., "Fe-Mn")
            valid_only: If True, only return valid structures
            source: Optional source filter ('ggen', 'mp', etc.)

        Returns:
            Dict mapping formula -> best StoredStructure
        """
        where, params = self._subsystem_clause(subsystem)
        where += " AND c.energy_per_atom IS NOT NULL"
        if valid_only:
            where += " AND s.is_valid = 1"
        if source:
            where += " AND s.source = ?"
            params.append(source)
        # Rank in SQL so only the winners' geometries are read.
        ranked = f"""
            s.id IN (
                SELECT id FROM (
                    SELECT s.id, ROW_NUMBER() OVER (
                        PARTITION BY s.formula ORDER BY c.energy_per_atom
                    ) AS rank
                    FROM structures s
                    JOIN calculations c ON c.structure_id = s.id AND c.surface = :surface
                    WHERE {where}
                ) WHERE rank = 1
            )
        """
        best = self.query_structures(ranked, params)
        return {s.formula: s for s in best}

    def _row_to_structure(self, row: sqlite3.Row) -> StoredStructure:
        """Convert a row selected with ``_STRUCTURE_COLUMNS`` to a StoredStructure."""
        keys = row.keys()
        return StoredStructure(
            id=row["id"],
            formula=row["formula"],
            elements=json.loads(row["elements"]),
            chemsys=row["chemsys"],
            stoichiometry=json.loads(row["stoichiometry"]),
            energy_per_atom=row["energy_per_atom"],
            total_energy=row["total_energy"],
            num_atoms=row["num_atoms"],
            space_group_number=row["space_group_number"],
            space_group_symbol=row["space_group_symbol"],
            structure_hash=row["structure_hash"],
            cif_content=decompress_cif(row["cif"]) if "cif" in keys else None,
            is_valid=bool(row["is_valid"]),
            error_message=row["error_message"],
            generation_metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            source=row["source"] or "ggen",
            source_id=row["source_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            e_above_hull=row["e_above_hull"],
            is_on_hull=bool(row["is_on_hull"]),
            is_dynamically_stable=_bool_or_none(row["is_dynamically_stable"]),
            num_imaginary_modes=row["num_imaginary_modes"],
            min_phonon_frequency=row["min_phonon_frequency"],
            max_phonon_frequency=row["max_phonon_frequency"],
            phonon_supercell=row["phonon_supercell"],
            reduced_formula=row["reduced_formula"],
            surface=self.surface,
            correction_per_atom=row["correction_per_atom"],
        )

    # -------------------- Hull Computation --------------------

    def hull_rows(self, chemical_system: str) -> List[sqlite3.Row]:
        """Energies on this surface for every valid structure in a system.

        One row per structure with an energy: the lean input to a hull,
        without geometries.
        """
        where, params = self._subsystem_clause(chemical_system)
        return self.conn.execute(
            f"""
            SELECT s.id, s.formula, s.reduced_formula, s.source, s.source_id,
                   s.space_group_symbol, c.energy_per_atom, c.correction_per_atom,
                   c.is_dynamically_stable
            FROM structures s
            JOIN calculations c ON c.structure_id = s.id AND c.surface = ?
            WHERE {where} AND s.is_valid = 1 AND c.energy_per_atom IS NOT NULL
            """,
            [self.surface, *params],
        ).fetchall()

    @staticmethod
    def hull_entry(
        row: Union[sqlite3.Row, Dict[str, Any]], entry_id: Optional[str] = None
    ) -> ComputedEntry:
        """A ComputedEntry for a hull row, labeled for phase diagram plots.

        The default ``entry_id`` is a readable label; pass the source id
        instead when a viewer should link entries (``mp-…`` opens Materials
        Project).
        """
        comp = Composition(row["formula"])
        sg = row["space_group_symbol"] or "?"
        source = (row["source"] or "ggen").lower()
        corrected = row["energy_per_atom"] + row["correction_per_atom"]
        return ComputedEntry(
            comp,
            corrected * comp.num_atoms,
            entry_id=entry_id or f"{row['formula']} ({sg.replace('/', '-')}, {source})",
            data={
                "space_group": row["space_group_symbol"],
                "structure_id": row["id"],
                "source": source,
                "source_id": row["source_id"],
            },
        )

    def compute_hull(
        self,
        chemical_system: str,
        update_database: bool = True,
        exclude_space_groups: Optional[List[str]] = None,
        tested_only: bool = False,
        all_polymorphs: bool = False,
    ) -> Tuple[Optional[PhaseDiagram], Dict[str, float], Dict[str, int]]:
        """
        Compute the convex hull for a chemical system on this surface.

        The hull is built from the lowest corrected energy of each reduced
        formula (every structure with ``all_polymorphs``), and every structure
        with an energy gets an ``e_above_hull``.

        Args:
            chemical_system: Chemical system (e.g., "Fe-Mn-Co")
            update_database: If True, store each structure's e_above_hull
            exclude_space_groups: List of space group symbols to exclude (e.g., ["P1"])
            tested_only: If True, only include structures that have been phonon-tested
            all_polymorphs: If True, put every structure in the phase diagram

        Returns:
            Tuple of (PhaseDiagram, dict mapping structure_id -> e_above_hull, filter_counts)
            filter_counts has keys like 'excluded_space_groups', 'excluded_untested',
            'excluded_incompatible'
        """
        chemsys = self.normalize_chemsys(chemical_system)
        filter_counts: Dict[str, int] = {}
        rows = list(self.hull_rows(chemsys))

        def drop(predicate, key: str) -> None:
            nonlocal rows
            kept = [r for r in rows if not predicate(r)]
            if len(kept) < len(rows):
                filter_counts[key] = len(rows) - len(kept)
            rows = kept

        # MP2020 rejects a few entries (and a correction is missing only if
        # the geometry was unreadable); either way they cannot share the hull.
        drop(lambda r: r["correction_per_atom"] is None, "excluded_incompatible")
        if exclude_space_groups:
            excluded = set(exclude_space_groups)
            drop(lambda r: r["space_group_symbol"] in excluded, "excluded_space_groups")
        if tested_only:
            drop(lambda r: r["is_dynamically_stable"] is None, "excluded_untested")

        if not rows:
            logger.warning(f"No structures found for {chemsys}")
            return None, {}, filter_counts

        entries = [(row, self.hull_entry(row)) for row in rows]
        if all_polymorphs:
            pd_entries = [entry for _, entry in entries]
        else:
            best: Dict[str, ComputedEntry] = {}
            for row, entry in entries:
                current = best.get(row["reduced_formula"])
                if current is None or entry.energy_per_atom < current.energy_per_atom:
                    best[row["reduced_formula"]] = entry
            pd_entries = list(best.values())

        if len(pd_entries) < 2:
            logger.warning(f"Need at least 2 entries for hull, got {len(pd_entries)}")
            return None, {}, filter_counts

        logger.info(
            f"Computing hull for {chemsys} on {self.surface} with "
            f"{len(pd_entries)} phases ({len(entries)} structures)"
        )
        try:
            pd = PhaseDiagram(pd_entries)
        except Exception as e:
            logger.error(f"Failed to build phase diagram for {chemsys}: {e}")
            return None, {}, filter_counts

        e_above_hull_map: Dict[str, float] = {}
        for row, entry in entries:
            try:
                e_hull = pd.get_e_above_hull(entry, allow_negative=True)
            except Exception as exc:
                logger.debug(f"No hull distance for {row['id']}: {exc}")
                continue
            if e_hull is not None:
                e_above_hull_map[row["id"]] = max(float(e_hull), 0.0)

        if update_database and not all_polymorphs and not self.readonly:
            now = _now()
            self.conn.executemany(
                """
                UPDATE calculations
                SET e_above_hull = ?, is_on_hull = ?, hull_computed_at = ?
                WHERE structure_id = ? AND surface = ?
                """,
                (
                    (e, 1 if e < ON_HULL_TOLERANCE else 0, now, sid, self.surface)
                    for sid, e in e_above_hull_map.items()
                ),
            )
            self._commit()
            on_hull = sum(1 for e in e_above_hull_map.values() if e < ON_HULL_TOLERANCE)
            logger.info(f"Updated hull for {chemsys}: {on_hull} entries on hull")

        return pd, e_above_hull_map, filter_counts

    def get_hull_entries(
        self,
        chemical_system: str,
        e_above_hull_cutoff: float = 0.0,
        run_id: Optional[str] = None,
    ) -> List[StoredStructure]:
        """
        Get structures on or near the convex hull, one per reduced formula.

        The hull is computed from current data rather than read from stored
        distances: a stored distance goes stale as soon as a subsystem gains
        a lower phase from a run in another system.

        Args:
            chemical_system: Chemical system (e.g., "Fe-Mn-Co")
            e_above_hull_cutoff: Maximum e_above_hull in eV/atom (0 = only hull entries)
            run_id: If provided, only return structures linked to this run.
                Hull distances are still computed from all data, but only
                structures generated in the given run are returned, and only
                when they are the best of their formula.

        Returns:
            List of StoredStructure objects on/near hull, sorted by e_above_hull
        """
        _, e_hull, _ = self.compute_hull(chemical_system, update_database=False)
        cutoff = e_above_hull_cutoff + ON_HULL_TOLERANCE
        near = [(e, sid) for sid, e in e_hull.items() if e <= cutoff]
        if not near:
            return []
        ids = [sid for _, sid in near]
        loaded = {
            s.id: s
            for start in range(0, len(ids), 500)
            for s in self.query_structures(
                f"s.id IN ({','.join('?' * len(ids[start:start + 500]))})",
                ids[start : start + 500],
            )
        }
        best: Dict[str, StoredStructure] = {}
        for e, sid in sorted(near, key=lambda item: (item[0], loaded[item[1]].energy_per_atom)):
            s = loaded[sid]
            s.e_above_hull = e
            s.is_on_hull = e < ON_HULL_TOLERANCE
            best.setdefault(s.reduced_formula, s)
        structures = list(best.values())
        if run_id is not None:
            in_run = {
                row[0]
                for row in self.conn.execute(
                    "SELECT structure_id FROM run_structures WHERE run_id = ?",
                    (run_id,),
                )
            }
            structures = [s for s in structures if s.id in in_run]
        return structures

    # -------------------- Novelty --------------------

    def get_match_candidates(self, reduced_formula: str) -> List["Candidate"]:
        """Every valid structure with a reduced formula, with one geometry each.

        The published geometry is preferred; GGen and user structures only
        have the one they were relaxed to.
        """
        from .matching import Candidate

        rows = self.conn.execute(
            """
            SELECT s.id, s.source, s.source_id, s.formula, s.space_group_number,
                COALESCE(
                    (SELECT g.cif FROM geometries g
                     WHERE g.structure_id = s.id AND g.kind = 'source'),
                    (SELECT g.cif FROM geometries g
                     WHERE g.structure_id = s.id ORDER BY g.kind LIMIT 1)
                ) AS cif
            FROM structures s
            WHERE s.reduced_formula = ? AND s.is_valid = 1
            """,
            (reduced_formula,),
        ).fetchall()
        return [
            Candidate(
                structure_id=row["id"],
                source=row["source"],
                source_id=row["source_id"],
                formula=row["formula"],
                space_group_number=row["space_group_number"],
                cif=decompress_cif(row["cif"]),
            )
            for row in rows
            if row["cif"] is not None
        ]

    def novelty_index(self) -> "NoveltyIndex":
        """A NoveltyIndex over this database (see ``ggen.matching``)."""
        from .matching import NoveltyIndex

        return NoveltyIndex(self)

    # -------------------- Run Management --------------------

    def create_run(
        self,
        chemical_system: str,
        parameters: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Create a new exploration run.

        Returns:
            Run ID (UUID)
        """
        run_id = str(uuid.uuid4())
        chemsys = self.normalize_chemsys(chemical_system)
        elements = chemsys.split("-")
        self.conn.execute(
            """
            INSERT INTO runs (id, chemical_system, elements, surface, parameters, started_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                chemsys,
                json.dumps(elements),
                self.surface,
                json.dumps(parameters) if parameters else None,
                _now(),
                notes,
            ),
        )
        self._commit()

        logger.info(f"Created run {run_id[:8]}... for {chemsys}")
        return run_id

    def complete_run(
        self,
        run_id: str,
        num_candidates: int = 0,
        num_successful: int = 0,
        num_failed: int = 0,
    ) -> None:
        """Mark a run as completed."""
        self.conn.execute(
            """
            UPDATE runs SET
                completed_at = ?,
                num_candidates = ?,
                num_successful = ?,
                num_failed = ?
            WHERE id = ?
            """,
            (_now(), num_candidates, num_successful, num_failed, run_id),
        )
        self._commit()

    def link_structure_to_run(
        self,
        run_id: str,
        structure_id: str,
        e_above_hull: Optional[float] = None,
        is_on_hull: bool = False,
    ) -> None:
        """Link a structure to an exploration run."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO run_structures
            (run_id, structure_id, e_above_hull, is_on_hull)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, structure_id, e_above_hull, 1 if is_on_hull else 0),
        )
        self._commit()

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> ExplorationRun:
        return ExplorationRun(
            id=row["id"],
            chemical_system=row["chemical_system"],
            elements=json.loads(row["elements"]),
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            num_candidates=row["num_candidates"],
            num_successful=row["num_successful"],
            num_failed=row["num_failed"],
            notes=row["notes"],
        )

    def get_run(self, run_id: str) -> Optional[ExplorationRun]:
        """Get a run by ID."""
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return None if row is None else self._row_to_run(row)

    def get_runs_for_system(self, chemical_system: str) -> List[ExplorationRun]:
        """Get all runs for a chemical system."""
        rows = self.conn.execute(
            "SELECT * FROM runs WHERE chemical_system = ? ORDER BY started_at DESC",
            (self.normalize_chemsys(chemical_system),),
        ).fetchall()
        return [self._row_to_run(row) for row in rows]

    # -------------------- Statistics & Queries --------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics, including breakdown by source."""
        conn = self.conn
        stats: Dict[str, Any] = {}
        stats["total_structures"] = conn.execute(
            "SELECT COUNT(*) FROM structures"
        ).fetchone()[0]
        stats["valid_structures"] = conn.execute(
            "SELECT COUNT(*) FROM structures WHERE is_valid = 1"
        ).fetchone()[0]
        stats["total_runs"] = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        stats["unique_formulas"] = conn.execute(
            "SELECT COUNT(DISTINCT reduced_formula) FROM structures WHERE is_valid = 1"
        ).fetchone()[0]
        stats["unique_chemsys"] = conn.execute(
            "SELECT COUNT(DISTINCT chemsys) FROM structures WHERE is_valid = 1"
        ).fetchone()[0]
        stats["by_source"] = {
            row[0]: row[1]
            for row in conn.execute(
                """
                SELECT source, COUNT(*) AS cnt FROM structures WHERE is_valid = 1
                GROUP BY source ORDER BY cnt DESC
                """
            )
        }
        stats["by_surface"] = {
            row[0]: row[1]
            for row in conn.execute(
                """
                SELECT surface, COUNT(*) FROM calculations
                WHERE energy_per_atom IS NOT NULL GROUP BY surface
                """
            )
        }
        # Valid structures with no energy on this surface yet
        stats["unrelaxed"] = stats["valid_structures"] - conn.execute(
            """
            SELECT COUNT(*) FROM calculations c JOIN structures s ON s.id = c.structure_id
            WHERE c.surface = ? AND c.energy_per_atom IS NOT NULL AND s.is_valid = 1
            """,
            (self.surface,),
        ).fetchone()[0]
        stats["explored_systems"] = self.list_explored_systems()
        return stats

    def get_system_statistics(
        self, chemical_systems: List[str], e_above_hull_cutoff: float = 0.05
    ) -> Dict[str, Any]:
        """Summarize everything the database knows about one or more systems.

        Counts cover each system and all of its subsystems, since a Bi-Fe-S hull
        is shaped by the Bi, Fe, S, Bi-Fe, … structures as well. Structures are
        counted once even when the systems share subsystems, while hull and run
        counts are summed across them.
        """
        chemsys_list = [self.normalize_chemsys(system) for system in chemical_systems]
        subsystems = sorted(
            {
                subsystem
                for chemsys in chemsys_list
                for subsystem in self.get_all_subsystems(chemsys.split("-"))
            }
        )
        placeholders = ",".join("?" * len(subsystems))

        totals = self.conn.execute(
            f"""
            SELECT COUNT(*) AS structures,
                   COUNT(DISTINCT formula) AS formulas,
                   COUNT(DISTINCT chemsys) AS subsystems
            FROM structures
            WHERE is_valid = 1 AND chemsys IN ({placeholders})
            """,
            subsystems,
        ).fetchone()

        by_source = {
            row["source"]: row["cnt"]
            for row in self.conn.execute(
                f"""
                SELECT source, COUNT(*) AS cnt
                FROM structures
                WHERE is_valid = 1 AND chemsys IN ({placeholders})
                GROUP BY source ORDER BY cnt DESC
                """,
                subsystems,
            ).fetchall()
        }

        # Terminal elements are on the hull by construction, so the hull counts
        # only cover compounds, one per reduced formula. Each hull is per
        # system, hence the summation.
        on_hull = near_hull = 0
        for chemsys in chemsys_list:
            subs = self.get_all_subsystems(chemsys.split("-"))
            row = self.conn.execute(
                f"""
                SELECT SUM(best < ?) AS on_hull, SUM(best <= ?) AS near_hull FROM (
                    SELECT MIN(c.e_above_hull) AS best
                    FROM structures s
                    JOIN calculations c ON c.structure_id = s.id AND c.surface = ?
                    WHERE s.is_valid = 1 AND s.chemsys LIKE '%-%'
                      AND s.chemsys IN ({','.join('?' * len(subs))})
                      AND c.e_above_hull IS NOT NULL
                    GROUP BY s.reduced_formula
                )
                """,
                [ON_HULL_TOLERANCE, e_above_hull_cutoff, self.surface, *subs],
            ).fetchone()
            on_hull += row["on_hull"] or 0
            near_hull += row["near_hull"] or 0

        system_placeholders = ",".join("?" * len(chemsys_list))
        runs = self.conn.execute(
            f"""
            SELECT COUNT(*) AS total, MAX(completed_at) AS last_completed_at
            FROM runs WHERE chemical_system IN ({system_placeholders})
            """,
            chemsys_list,
        ).fetchone()

        return {
            "chemsys": chemsys_list,
            "total_structures": totals["structures"],
            "unique_formulas": totals["formulas"],
            "subsystems_covered": totals["subsystems"],
            "by_source": by_source,
            "on_hull": on_hull,
            "near_hull": near_hull,
            "e_above_hull_cutoff": e_above_hull_cutoff,
            "total_runs": runs["total"],
            "last_run_at": runs["last_completed_at"],
        }

    def get_run_statistics(self, run_id: str) -> Dict[str, Any]:
        """Count the structures a single exploration run contributed."""
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS structures, COUNT(DISTINCT s.formula) AS formulas
            FROM run_structures rs
            JOIN structures s ON s.id = rs.structure_id
            WHERE rs.run_id = ? AND s.is_valid = 1
            """,
            (run_id,),
        ).fetchone()
        return {
            "structures": row["structures"],
            "unique_formulas": row["formulas"],
        }

    def list_explored_systems(self) -> List[str]:
        """Get list of all explored chemical systems."""
        rows = self.conn.execute(
            "SELECT DISTINCT chemical_system FROM runs ORDER BY chemical_system"
        ).fetchall()
        return [row[0] for row in rows]

    def list_formulas(self, chemical_system: Optional[str] = None) -> List[str]:
        """Get list of all formulas, optionally filtered by chemical system."""
        if chemical_system:
            rows = self.conn.execute(
                """
                SELECT DISTINCT formula FROM structures
                WHERE chemsys = ? AND is_valid = 1
                ORDER BY formula
                """,
                (self.normalize_chemsys(chemical_system),),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT DISTINCT formula FROM structures WHERE is_valid = 1 ORDER BY formula"
            ).fetchall()
        return [row[0] for row in rows]

    def search_structures(
        self,
        formula_pattern: Optional[str] = None,
        elements: Optional[List[str]] = None,
        max_energy_per_atom: Optional[float] = None,
        space_group: Optional[int] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[StoredStructure]:
        """
        Search structures with various filters.

        Args:
            formula_pattern: SQL LIKE pattern for formula (e.g., "Fe%Mn%")
            elements: List of elements that must all be present
            max_energy_per_atom: Maximum raw energy per atom on this surface
            space_group: Required space group number
            source: Optional source filter ('ggen', 'mp', etc.)
            limit: Maximum results to return

        Returns:
            List of matching StoredStructure objects
        """
        conditions = ["s.is_valid = 1"]
        params: List[Any] = []

        if formula_pattern:
            conditions.append("s.formula LIKE ?")
            params.append(formula_pattern)
        if max_energy_per_atom is not None:
            conditions.append("c.energy_per_atom <= ?")
            params.append(max_energy_per_atom)
        if space_group is not None:
            conditions.append("s.space_group_number = ?")
            params.append(space_group)
        if source is not None:
            conditions.append("s.source = ?")
            params.append(source)
        for element in elements or []:
            conditions.append("EXISTS (SELECT 1 FROM json_each(s.elements) WHERE value = ?)")
            params.append(element)

        return self.query_structures(
            " AND ".join(conditions),
            params,
            order_by="c.energy_per_atom IS NULL, c.energy_per_atom",
            limit=limit,
        )

    # -------------------- Legacy run directories --------------------

    def import_exploration_run(
        self,
        run_directory: Union[str, Path],
        recompute_hulls: bool = True,
    ) -> Tuple[int, int]:
        """
        Import structures from an existing exploration run directory.

        Args:
            run_directory: Path to the exploration run directory (contains exploration.db)
            recompute_hulls: If True, recompute hulls for affected chemical systems

        Returns:
            Tuple of (structures_imported, structures_skipped)
        """
        run_dir = Path(run_directory)
        db_path = run_dir / "exploration.db"

        if not db_path.exists():
            raise ValueError(f"No database found at {db_path}")

        run_conn = sqlite3.connect(str(db_path))
        run_conn.row_factory = sqlite3.Row

        metadata = {}
        for row in run_conn.execute("SELECT key, value FROM run_metadata"):
            metadata[row["key"]] = row["value"]

        chemical_system = metadata.get("chemical_system", "unknown")
        started_at = metadata.get("started_at")
        completed_at = metadata.get("completed_at")

        run_id = self.create_run(
            chemical_system=chemical_system,
            parameters={
                "imported_from": str(run_dir),
                "original_metadata": metadata,
            },
            notes=f"Imported from {run_dir.name}",
        )

        if started_at:
            self.conn.execute(
                "UPDATE runs SET started_at = ? WHERE id = ?", (started_at, run_id)
            )
        if completed_at:
            self.conn.execute(
                "UPDATE runs SET completed_at = ? WHERE id = ?", (completed_at, run_id)
            )
        self._commit()

        structures_dir = run_dir / "structures"
        imported = 0
        skipped = 0
        affected_systems: set = set()

        for row in run_conn.execute("SELECT * FROM candidates WHERE is_valid = 1"):
            try:
                formula = row["formula"]
                stoichiometry = json.loads(row["stoichiometry"])
                energy_per_atom = row["energy_per_atom"]
                total_energy = row["total_energy"]
                num_atoms = row["num_atoms"]
                space_group_number = row["space_group_number"]
                space_group_symbol = row["space_group_symbol"]
                cif_filename = row["cif_filename"]
                generation_metadata = (
                    json.loads(row["generation_metadata"])
                    if row["generation_metadata"]
                    else {}
                )

                if energy_per_atom is None:
                    skipped += 1
                    continue

                cif_content = None
                if cif_filename and structures_dir.exists():
                    cif_path = structures_dir / cif_filename
                    if cif_path.exists():
                        cif_content = cif_path.read_text()

                self.add_structure(
                    formula=formula,
                    stoichiometry=stoichiometry,
                    energy_per_atom=float(energy_per_atom),
                    total_energy=(
                        float(total_energy)
                        if total_energy
                        else energy_per_atom * num_atoms
                    ),
                    num_atoms=int(num_atoms),
                    space_group_number=(
                        int(space_group_number) if space_group_number else None
                    ),
                    space_group_symbol=space_group_symbol,
                    cif_content=cif_content,
                    is_valid=True,
                    generation_metadata={
                        **generation_metadata,
                        "imported_from": str(run_dir),
                    },
                    run_id=run_id,
                )
                imported += 1
                affected_systems.add("-".join(sorted(stoichiometry.keys())))

            except Exception as e:
                logger.warning(f"Failed to import {row['formula']}: {e}")
                skipped += 1
                continue

        run_conn.close()

        self.complete_run(
            run_id=run_id,
            num_candidates=imported + skipped,
            num_successful=imported,
            num_failed=skipped,
        )

        logger.info(
            f"Imported {imported} structures from {run_dir.name} "
            f"({skipped} skipped)"
        )

        if recompute_hulls:
            for chemsys in affected_systems:
                try:
                    self.compute_hull(chemsys)
                    logger.info(f"Recomputed hull for {chemsys}")
                except Exception as e:
                    logger.warning(f"Failed to compute hull for {chemsys}: {e}")

        return imported, skipped

    def import_all_runs(
        self,
        base_directory: Union[str, Path],
        pattern: str = "exploration_*",
        recompute_hulls: bool = True,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Import all exploration runs from a directory.

        Args:
            base_directory: Directory containing exploration run folders
            pattern: Glob pattern to match run directories
            recompute_hulls: If True, recompute hulls after importing all runs

        Returns:
            Dictionary mapping run_directory -> (imported, skipped)
        """
        base_dir = Path(base_directory)
        results: Dict[str, Tuple[int, int]] = {}
        affected_systems: set = set()

        run_dirs = sorted(base_dir.glob(pattern))
        logger.info(f"Found {len(run_dirs)} exploration runs to import")

        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            if not (run_dir / "exploration.db").exists():
                continue

            try:
                imported, skipped = self.import_exploration_run(
                    run_dir, recompute_hulls=False
                )
                results[str(run_dir)] = (imported, skipped)

                name = run_dir.name
                if name.startswith("exploration_"):
                    parts = name.split("_")
                    if len(parts) >= 2:
                        affected_systems.add(parts[1])

            except Exception as e:
                logger.error(f"Failed to import {run_dir}: {e}")
                results[str(run_dir)] = (0, 0)

        if recompute_hulls:
            logger.info(
                f"Recomputing hulls for {len(affected_systems)} chemical systems"
            )
            for chemsys in sorted(affected_systems):
                try:
                    self.compute_hull(chemsys)
                except Exception as e:
                    logger.warning(f"Failed to compute hull for {chemsys}: {e}")

        total_imported = sum(r[0] for r in results.values())
        total_skipped = sum(r[1] for r in results.values())
        logger.info(
            f"Migration complete: {total_imported} structures imported, "
            f"{total_skipped} skipped from {len(results)} runs"
        )

        return results

    # -------------------- Shards: extract, export, merge --------------------
    #
    # Only one process may write the shared file (a Modal Volume keeps the
    # last commit of a file, so two writers lose each other's rows). Workers
    # instead extract the systems they need into a local file, work on it,
    # export what changed, and hand that shard to the single writer, which
    # merges it with ``merge_from``.

    def _copy_into(self, alias: str, id_table: str) -> int:
        """Copy the structures listed in temp table ``id_table`` into ``alias``."""
        conn = self.conn
        for table in ("structures", "geometries", "calculations"):
            cols = [
                c for c in self._table_columns(table) if c in self._table_columns(table, alias)
            ]
            collist = ", ".join(cols)
            key = "id" if table == "structures" else "structure_id"
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {alias}.{table} ({collist})
                SELECT {collist} FROM main.{table}
                WHERE {key} IN (SELECT id FROM temp.{id_table})
                """
            )
        return conn.execute(f"SELECT COUNT(*) FROM {alias}.structures").fetchone()[0]

    def _with_attached(self, out_path: Path, fill: Callable[[str], int]) -> int:
        if out_path.exists():
            out_path.unlink()
        StructureDatabase(out_path, surface=self.surface).close()
        conn = self.conn
        # A read-only database (opened mode=ro) stays read-only; query_only
        # only has to be lifted so the temp table and the shard can be written.
        if self.readonly:
            conn.execute("PRAGMA query_only=OFF")
        conn.execute("ATTACH DATABASE ? AS shard", (str(out_path),))
        try:
            count = fill("shard")
            conn.commit()
        finally:
            conn.execute("DROP TABLE IF EXISTS temp._shard_ids")
            conn.execute("DETACH DATABASE shard")
            if self.readonly:
                conn.execute("PRAGMA query_only=ON")
        return count

    def extract_subset(self, out_path: Union[str, Path], structure_ids: List[str]) -> int:
        """Write the given structures (all surfaces, all geometries) to a fresh DB."""

        def fill(alias: str) -> int:
            self.conn.execute("CREATE TEMP TABLE _shard_ids (id TEXT PRIMARY KEY)")
            self.conn.executemany(
                "INSERT OR IGNORE INTO temp._shard_ids (id) VALUES (?)",
                ((sid,) for sid in structure_ids),
            )
            return self._copy_into(alias, "_shard_ids")

        return self._with_attached(Path(out_path), fill)

    def extract_systems(
        self, out_path: Union[str, Path], chemical_systems: Iterable[str]
    ) -> int:
        """Write every structure in the systems (and their subsystems) to a fresh DB.

        This is a worker's local working copy: everything a hull over those
        systems needs, plus the runs that touched them.
        """
        subsystems = sorted(
            {
                sub
                for system in chemical_systems
                for sub in self.get_all_subsystems(
                    self.normalize_chemsys(system).split("-")
                )
            }
        )

        def fill(alias: str) -> int:
            conn = self.conn
            conn.execute(
                f"""
                CREATE TEMP TABLE _shard_ids AS
                SELECT id FROM main.structures
                WHERE chemsys IN ({','.join('?' * len(subsystems))})
                """,
                subsystems,
            )
            count = self._copy_into(alias, "_shard_ids")
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {alias}.run_structures
                SELECT * FROM main.run_structures
                WHERE structure_id IN (SELECT id FROM temp._shard_ids)
                """
            )
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {alias}.runs
                SELECT * FROM main.runs
                WHERE id IN (SELECT run_id FROM {alias}.run_structures)
                """
            )
            return count

        return self._with_attached(Path(out_path), fill)

    def export_changes(self, out_path: Union[str, Path], since: str) -> int:
        """Write everything created or changed at or after ``since`` to a fresh DB.

        ``since`` is an ISO timestamp taken before the work started. Returns
        the number of structures exported.
        """

        def fill(alias: str) -> int:
            conn = self.conn
            conn.execute(
                """
                CREATE TEMP TABLE _shard_ids AS
                SELECT id FROM main.structures WHERE updated_at >= ?
                UNION
                SELECT structure_id FROM main.calculations WHERE updated_at >= ?
                """,
                (since, since),
            )
            count = self._copy_into(alias, "_shard_ids")
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {alias}.runs
                SELECT * FROM main.runs
                WHERE started_at >= ? OR completed_at >= ?
                """,
                (since, since),
            )
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {alias}.run_structures
                SELECT * FROM main.run_structures
                WHERE run_id IN (SELECT id FROM {alias}.runs)
                   OR structure_id IN (SELECT id FROM temp._shard_ids)
                """
            )
            return count

        return self._with_attached(Path(out_path), fill)

    def merge_from(
        self, other_db_path: Union[str, Path], recompute_hulls: bool = True
    ) -> Tuple[int, int]:
        """Merge a shard (from ``export_changes`` or ``extract_subset``) into this DB.

        Structures are matched by id, then by (source, source_id), then by
        structure_hash, so a re-import or a duplicate GGen structure lands on
        the existing row. Matched structures take the shard's metadata,
        geometries and calculations when the shard's are newer. Runs and run
        links are copied with ids remapped. Hulls of affected systems are
        recomputed on this database's surface.

        Returns ``(new_structures, merged_into_existing)``.
        """
        conn = self.conn
        conn.execute("ATTACH DATABASE ? AS shard", (str(other_db_path),))
        try:
            conn.execute("DROP TABLE IF EXISTS temp._merge_map")
            conn.execute(
                """
                CREATE TEMP TABLE _merge_map (
                    shard_id TEXT PRIMARY KEY,
                    target_id TEXT NOT NULL,
                    is_new INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO temp._merge_map (shard_id, target_id, is_new)
                SELECT sh.id,
                    COALESCE(
                        (SELECT m.id FROM main.structures m WHERE m.id = sh.id),
                        (SELECT m.id FROM main.structures m
                         WHERE sh.source_id IS NOT NULL
                           AND m.source = sh.source AND m.source_id = sh.source_id),
                        (SELECT m.id FROM main.structures m
                         WHERE sh.source_id IS NULL AND sh.structure_hash IS NOT NULL
                           AND m.structure_hash = sh.structure_hash
                         LIMIT 1),
                        sh.id
                    ),
                    0
                FROM shard.structures sh
                """
            )
            conn.execute(
                """
                UPDATE temp._merge_map SET is_new = 1
                WHERE target_id NOT IN (SELECT id FROM main.structures)
                """
            )

            s_cols = [c for c in self._table_columns("structures") if c != "id"]
            conn.execute(
                f"""
                INSERT INTO main.structures (id, {', '.join(s_cols)})
                SELECT mm.target_id, {', '.join('sh.' + c for c in s_cols)}
                FROM shard.structures sh JOIN temp._merge_map mm ON mm.shard_id = sh.id
                WHERE mm.is_new = 1
                """
            )
            # Existing rows take newer identity fields (space group backfills,
            # metadata) but keep their own id and created_at.
            updatable = [
                c for c in s_cols if c not in ("created_at", "source", "source_id")
            ]
            conn.execute(
                f"""
                UPDATE main.structures AS m SET
                    {', '.join(f'{c} = sh.{c}' for c in updatable)}
                FROM shard.structures sh JOIN temp._merge_map mm ON mm.shard_id = sh.id
                WHERE mm.is_new = 0 AND m.id = mm.target_id
                  AND sh.updated_at > m.updated_at
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO main.geometries (structure_id, kind, cif)
                SELECT mm.target_id, g.kind, g.cif
                FROM shard.geometries g JOIN temp._merge_map mm ON mm.shard_id = g.structure_id
                """
            )
            c_cols = [
                c for c in self._table_columns("calculations") if c != "structure_id"
            ]
            conn.execute(
                f"""
                INSERT INTO main.calculations (structure_id, {', '.join(c_cols)})
                SELECT mm.target_id, {', '.join('c.' + c for c in c_cols)}
                FROM shard.calculations c JOIN temp._merge_map mm ON mm.shard_id = c.structure_id
                WHERE true
                ON CONFLICT (structure_id, surface) DO UPDATE SET
                    {', '.join(f'{c} = excluded.{c}' for c in c_cols if c not in ('surface', 'created_at'))}
                WHERE excluded.updated_at > calculations.updated_at
                """
            )
            conn.execute("INSERT OR REPLACE INTO main.runs SELECT * FROM shard.runs")
            conn.execute(
                """
                INSERT OR REPLACE INTO main.run_structures
                    (run_id, structure_id, e_above_hull, is_on_hull)
                SELECT rs.run_id, COALESCE(mm.target_id, rs.structure_id),
                       rs.e_above_hull, rs.is_on_hull
                FROM shard.run_structures rs
                LEFT JOIN temp._merge_map mm ON mm.shard_id = rs.structure_id
                """
            )
            new = conn.execute(
                "SELECT COUNT(*) FROM temp._merge_map WHERE is_new = 1"
            ).fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM temp._merge_map").fetchone()[0]
            affected = [
                row[0]
                for row in conn.execute(
                    """
                    SELECT DISTINCT s.chemsys FROM main.structures s
                    WHERE s.id IN (SELECT target_id FROM temp._merge_map)
                    """
                )
            ]
            self._commit()
        finally:
            conn.execute("DROP TABLE IF EXISTS temp._merge_map")
            conn.execute("DETACH DATABASE shard")

        if recompute_hulls:
            # A hull over a system covers its subsystems, so only the largest
            # affected systems need computing.
            systems = set(affected)
            maximal = [
                s
                for s in systems
                if not any(
                    s != other and set(s.split("-")) < set(other.split("-"))
                    for other in systems
                )
            ]
            for chemsys in maximal:
                try:
                    self.compute_hull(chemsys)
                except Exception as exc:
                    logger.warning(f"Failed to compute hull for {chemsys}: {exc}")

        logger.info(
            f"Merged {Path(other_db_path).name}: {new} new structures, "
            f"{total - new} merged into existing"
        )
        return new, total - new
