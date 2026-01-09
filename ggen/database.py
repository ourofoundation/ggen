"""
Unified Structure Database for GGen

Provides persistent storage for crystal structures across exploration runs,
with automatic subsystem indexing and dynamic hull computation.
"""

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


@dataclass
class StoredStructure:
    """A structure stored in the unified database."""

    id: str
    formula: str
    elements: List[str]
    chemsys: str
    stoichiometry: Dict[str, int]

    energy_per_atom: float
    total_energy: float
    num_atoms: int

    space_group_number: Optional[int] = None
    space_group_symbol: Optional[str] = None

    structure_hash: Optional[str] = None
    cif_content: Optional[str] = None

    is_valid: bool = True
    error_message: Optional[str] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Hull info (computed dynamically)
    e_above_hull: Optional[float] = None
    is_on_hull: bool = False

    # Dynamical stability (phonon) info
    is_dynamically_stable: Optional[bool] = None
    num_imaginary_modes: Optional[int] = None
    min_phonon_frequency: Optional[float] = None
    max_phonon_frequency: Optional[float] = None
    phonon_supercell: Optional[str] = None  # JSON string of supercell tuple

    # Loaded structure object (not stored in DB)
    structure: Optional[Structure] = None

    def get_structure(self) -> Optional[Structure]:
        """Get the pymatgen Structure, loading from CIF if needed."""
        if self.structure is not None:
            return self.structure
        if self.cif_content:
            try:
                from io import StringIO

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


class StructureDatabase:
    """
    Unified database for storing and querying crystal structures.

    Key features:
    - Automatic subsystem indexing (Fe2CoMn indexed under Fe, Co, Mn, Fe-Co, etc.)
    - Dynamic hull computation across all known structures
    - Deduplication via structure hashing
    - Track exploration run history

    Example:
        >>> db = StructureDatabase("./ggen.db")
        >>> db.add_structure(formula="Fe3Mn", energy_per_atom=-8.5, ...)
        >>> best = db.get_best_structures("Fe-Mn")
        >>> db.compute_hull("Fe-Mn-Co")  # Uses all relevant structures
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize the structure database.

        Args:
            db_path: Path to SQLite database. Defaults to ./ggen.db in current directory.
        """
        if db_path is None:
            db_path = Path.cwd() / "ggen.db"
        else:
            db_path = Path(db_path).expanduser()

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path

        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection, creating if needed.

        Uses WAL mode and busy timeout for better concurrent access
        when multiple processes are writing to the same database.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=60.0,  # Wait up to 60s for locks
                isolation_level="DEFERRED",  # Don't lock until needed
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            # WAL allows readers to not block writers and vice versa
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=60000")  # 60s in milliseconds
            self._conn.execute(
                "PRAGMA synchronous=NORMAL"
            )  # Good balance of safety/speed
        return self._conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self.conn

        # Main structures table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS structures (
                id TEXT PRIMARY KEY,
                formula TEXT NOT NULL,
                elements TEXT NOT NULL,
                chemsys TEXT NOT NULL,
                stoichiometry TEXT NOT NULL,
                
                energy_per_atom REAL,
                total_energy REAL,
                num_atoms INTEGER,
                
                space_group_number INTEGER,
                space_group_symbol TEXT,
                
                structure_hash TEXT,
                cif_content TEXT,
                
                is_valid INTEGER DEFAULT 1,
                error_message TEXT,
                generation_metadata TEXT,
                
                -- Dynamical stability (phonon) fields
                is_dynamically_stable INTEGER,
                num_imaginary_modes INTEGER,
                min_phonon_frequency REAL,
                max_phonon_frequency REAL,
                phonon_supercell TEXT,
                
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Migration: Add dynamical stability columns if they don't exist
        # This allows existing databases to be upgraded
        try:
            conn.execute(
                "ALTER TABLE structures ADD COLUMN is_dynamically_stable INTEGER"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute(
                "ALTER TABLE structures ADD COLUMN num_imaginary_modes INTEGER"
            )
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE structures ADD COLUMN min_phonon_frequency REAL")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE structures ADD COLUMN max_phonon_frequency REAL")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE structures ADD COLUMN phonon_supercell TEXT")
        except sqlite3.OperationalError:
            pass

        # Subsystem index table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS structure_subsystems (
                structure_id TEXT NOT NULL,
                subsystem TEXT NOT NULL,
                PRIMARY KEY (structure_id, subsystem),
                FOREIGN KEY (structure_id) REFERENCES structures(id)
            )
        """
        )

        # Exploration runs table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                chemical_system TEXT NOT NULL,
                elements TEXT NOT NULL,
                parameters TEXT,
                
                started_at TEXT,
                completed_at TEXT,
                num_candidates INTEGER DEFAULT 0,
                num_successful INTEGER DEFAULT 0,
                num_failed INTEGER DEFAULT 0,
                
                notes TEXT
            )
        """
        )

        # Run-structure links with hull info
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_structures (
                run_id TEXT NOT NULL,
                structure_id TEXT NOT NULL,
                e_above_hull REAL,
                is_on_hull INTEGER DEFAULT 0,
                PRIMARY KEY (run_id, structure_id),
                FOREIGN KEY (run_id) REFERENCES runs(id),
                FOREIGN KEY (structure_id) REFERENCES structures(id)
            )
        """
        )

        # Hull snapshots - track global hull state over time
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hull_snapshots (
                id TEXT PRIMARY KEY,
                chemsys TEXT NOT NULL,
                computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                num_entries INTEGER,
                hull_entries TEXT,
                metadata TEXT
            )
        """
        )

        # Hull entries - current global e_above_hull for each structure
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hull_entries (
                chemsys TEXT NOT NULL,
                structure_id TEXT NOT NULL,
                e_above_hull REAL,
                is_on_hull INTEGER DEFAULT 0,
                computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chemsys, structure_id),
                FOREIGN KEY (structure_id) REFERENCES structures(id)
            )
        """
        )

        # Indexes for fast queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_subsystem ON structure_subsystems(subsystem)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_formula ON structures(formula)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chemsys ON structures(chemsys)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_structure_hash ON structures(structure_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hull_chemsys ON hull_entries(chemsys)"
        )

        conn.commit()
        logger.info(f"Initialized database at {self.db_path}")

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @_retry_on_lock(max_retries=5, base_delay=0.1, max_delay=2.0)
    def _commit(self) -> None:
        """Commit with retry on lock errors."""
        self.conn.commit()

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
        elements = [e.strip() for e in system.replace("-", " ").split()]
        return "-".join(sorted(elements))

    @staticmethod
    def compute_structure_hash(
        formula: str, energy_per_atom: float, space_group: Optional[int] = None
    ) -> str:
        """Compute a hash for structure deduplication."""
        key = f"{formula}:{energy_per_atom:.6f}:{space_group or 'unknown'}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    # -------------------- Structure Operations --------------------

    def add_structure(
        self,
        formula: str,
        stoichiometry: Dict[str, int],
        energy_per_atom: float,
        total_energy: float,
        num_atoms: int,
        space_group_number: Optional[int] = None,
        space_group_symbol: Optional[str] = None,
        cif_content: Optional[str] = None,
        structure: Optional[Structure] = None,
        is_valid: bool = True,
        error_message: Optional[str] = None,
        generation_metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        # Phonon stability fields
        is_dynamically_stable: Optional[bool] = None,
        num_imaginary_modes: Optional[int] = None,
        min_phonon_frequency: Optional[float] = None,
        max_phonon_frequency: Optional[float] = None,
        phonon_supercell: Optional[Tuple[int, int, int]] = None,
    ) -> str:
        """
        Add a structure to the database.

        Args:
            formula: Chemical formula (e.g., "Fe3Mn")
            stoichiometry: Element counts {"Fe": 3, "Mn": 1}
            energy_per_atom: Energy per atom in eV
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
            is_dynamically_stable: Whether structure is dynamically stable (no imaginary phonon modes)
            num_imaginary_modes: Number of imaginary phonon modes
            min_phonon_frequency: Minimum phonon frequency in THz
            max_phonon_frequency: Maximum phonon frequency in THz
            phonon_supercell: Supercell dimensions used for phonon calculation

        Returns:
            Structure ID (UUID)
        """
        elements = sorted(stoichiometry.keys())
        chemsys = "-".join(elements)

        # Get CIF content from structure if not provided
        if cif_content is None and structure is not None:
            try:
                cif_content = structure.to(fmt="cif")
            except Exception as e:
                logger.warning(f"Failed to generate CIF for {formula}: {e}")

        # Compute hash for deduplication
        structure_hash = self.compute_structure_hash(
            formula, energy_per_atom, space_group_number
        )

        # Check for existing structure with same hash
        existing = self.conn.execute(
            "SELECT id, energy_per_atom FROM structures WHERE structure_hash = ?",
            (structure_hash,),
        ).fetchone()

        if existing:
            # Structure already exists - update if this one is better
            if energy_per_atom < existing["energy_per_atom"]:
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

        # Create new structure
        structure_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        self.conn.execute(
            """
            INSERT INTO structures (
                id, formula, elements, chemsys, stoichiometry,
                energy_per_atom, total_energy, num_atoms,
                space_group_number, space_group_symbol,
                structure_hash, cif_content,
                is_valid, error_message, generation_metadata,
                is_dynamically_stable, num_imaginary_modes,
                min_phonon_frequency, max_phonon_frequency, phonon_supercell,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                structure_id,
                formula,
                json.dumps(elements),
                chemsys,
                json.dumps(stoichiometry),
                energy_per_atom,
                total_energy,
                num_atoms,
                space_group_number,
                space_group_symbol,
                structure_hash,
                cif_content,
                1 if is_valid else 0,
                error_message,
                json.dumps(generation_metadata) if generation_metadata else None,
                (
                    1
                    if is_dynamically_stable
                    else (0 if is_dynamically_stable is False else None)
                ),
                num_imaginary_modes,
                min_phonon_frequency,
                max_phonon_frequency,
                json.dumps(phonon_supercell) if phonon_supercell else None,
                now,
                now,
            ),
        )

        # Index by all subsystems
        subsystems = self.get_all_subsystems(elements)
        for subsystem in subsystems:
            self.conn.execute(
                "INSERT OR IGNORE INTO structure_subsystems VALUES (?, ?)",
                (structure_id, subsystem),
            )

        self._commit()

        # Link to run if provided
        if run_id:
            self.link_structure_to_run(run_id, structure_id)

        logger.debug(f"Added structure {formula} (id={structure_id[:8]}...)")
        return structure_id

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
        """Update an existing structure with better data.

        Args:
            structure_id: ID of structure to update
            energy_per_atom: New energy per atom value
            total_energy: New total energy value
            cif_content: New CIF content
            generation_metadata: New metadata dict
            is_dynamically_stable: Phonon stability (True/False)
            num_imaginary_modes: Number of imaginary phonon modes
            min_phonon_frequency: Minimum phonon frequency
            max_phonon_frequency: Maximum phonon frequency
            phonon_supercell: Supercell used for phonon calculation
            clear_phonon_data: If True, set all phonon fields to NULL
        """
        updates = []
        values = []

        if energy_per_atom is not None:
            updates.append("energy_per_atom = ?")
            values.append(energy_per_atom)
        if total_energy is not None:
            updates.append("total_energy = ?")
            values.append(total_energy)
        if cif_content is not None:
            updates.append("cif_content = ?")
            values.append(cif_content)
        if generation_metadata is not None:
            updates.append("generation_metadata = ?")
            values.append(json.dumps(generation_metadata))

        # Handle phonon data - either clear all or update individually
        if clear_phonon_data:
            updates.append("is_dynamically_stable = NULL")
            updates.append("num_imaginary_modes = NULL")
            updates.append("min_phonon_frequency = NULL")
            updates.append("max_phonon_frequency = NULL")
            updates.append("phonon_supercell = NULL")
        else:
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
                values.append(json.dumps(phonon_supercell))

        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(structure_id)

        self.conn.execute(
            f"UPDATE structures SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        self._commit()

    def get_structure(self, structure_id: str) -> Optional[StoredStructure]:
        """Get a structure by ID."""
        row = self.conn.execute(
            "SELECT * FROM structures WHERE id = ?", (structure_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_structure(row)

    def get_structures_by_formula(
        self, formula: str, valid_only: bool = True
    ) -> List[StoredStructure]:
        """Get all structures with a given formula."""
        query = "SELECT * FROM structures WHERE formula = ?"
        if valid_only:
            query += " AND is_valid = 1"
        query += " ORDER BY energy_per_atom ASC"

        rows = self.conn.execute(query, (formula,)).fetchall()
        return [self._row_to_structure(row) for row in rows]

    def get_best_structure(
        self, formula: str, valid_only: bool = True
    ) -> Optional[StoredStructure]:
        """Get the lowest-energy structure for a formula."""
        structures = self.get_structures_by_formula(formula, valid_only)
        return structures[0] if structures else None

    def get_structures_for_subsystem(
        self, subsystem: str, valid_only: bool = True
    ) -> List[StoredStructure]:
        """
        Get all structures that belong to a chemical subsystem.

        For "Fe-Mn", returns Fe, Mn, and all Fe-Mn structures.
        """
        subsystem = self.normalize_chemsys(subsystem)
        elements = subsystem.split("-")

        # Get all valid subsystem chemsys values
        # For "Bi-Fe-S": ["Bi", "Fe", "S", "Bi-Fe", "Bi-S", "Fe-S", "Bi-Fe-S"]
        valid_chemsys = self.get_all_subsystems(elements)

        # Query structures whose chemsys is one of the valid subsystems
        placeholders = ",".join("?" * len(valid_chemsys))
        query = f"""
            SELECT * FROM structures
            WHERE chemsys IN ({placeholders})
        """
        if valid_only:
            query += " AND is_valid = 1"
        query += " ORDER BY formula, energy_per_atom"

        rows = self.conn.execute(query, valid_chemsys).fetchall()
        return [self._row_to_structure(row) for row in rows]

    def get_best_structures_for_subsystem(
        self, subsystem: str, valid_only: bool = True
    ) -> Dict[str, StoredStructure]:
        """
        Get the best (lowest energy) structure for each formula in a subsystem.

        Returns:
            Dict mapping formula -> best StoredStructure
        """
        structures = self.get_structures_for_subsystem(subsystem, valid_only)

        best_by_formula: Dict[str, StoredStructure] = {}
        for s in structures:
            if s.formula not in best_by_formula:
                best_by_formula[s.formula] = s
            elif s.energy_per_atom < best_by_formula[s.formula].energy_per_atom:
                best_by_formula[s.formula] = s

        return best_by_formula

    def _row_to_structure(self, row: sqlite3.Row) -> StoredStructure:
        """Convert a database row to StoredStructure."""
        # Get hull info if available
        hull_row = self.conn.execute(
            """
            SELECT e_above_hull, is_on_hull FROM hull_entries
            WHERE structure_id = ? ORDER BY computed_at DESC LIMIT 1
            """,
            (row["id"],),
        ).fetchone()

        # Extract phonon fields (may not exist in older databases)
        row_keys = row.keys()
        is_dynamically_stable = None
        if (
            "is_dynamically_stable" in row_keys
            and row["is_dynamically_stable"] is not None
        ):
            is_dynamically_stable = bool(row["is_dynamically_stable"])

        num_imaginary_modes = (
            row["num_imaginary_modes"] if "num_imaginary_modes" in row_keys else None
        )
        min_phonon_frequency = (
            row["min_phonon_frequency"] if "min_phonon_frequency" in row_keys else None
        )
        max_phonon_frequency = (
            row["max_phonon_frequency"] if "max_phonon_frequency" in row_keys else None
        )

        phonon_supercell = None
        if "phonon_supercell" in row_keys and row["phonon_supercell"]:
            phonon_supercell = row["phonon_supercell"]  # Already JSON string

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
            cif_content=row["cif_content"],
            is_valid=bool(row["is_valid"]),
            error_message=row["error_message"],
            generation_metadata=(
                json.loads(row["generation_metadata"])
                if row["generation_metadata"]
                else {}
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            e_above_hull=hull_row["e_above_hull"] if hull_row else None,
            is_on_hull=bool(hull_row["is_on_hull"]) if hull_row else False,
            is_dynamically_stable=is_dynamically_stable,
            num_imaginary_modes=num_imaginary_modes,
            min_phonon_frequency=min_phonon_frequency,
            max_phonon_frequency=max_phonon_frequency,
            phonon_supercell=phonon_supercell,
        )

    # -------------------- Hull Computation --------------------

    def compute_hull(
        self, chemical_system: str, update_database: bool = True
    ) -> Tuple[Optional[PhaseDiagram], Dict[str, float]]:
        """
        Compute the convex hull for a chemical system using all known structures.

        Args:
            chemical_system: Chemical system (e.g., "Fe-Mn-Co")
            update_database: If True, update hull_entries table with results

        Returns:
            Tuple of (PhaseDiagram, dict mapping formula -> e_above_hull)
        """
        chemsys = self.normalize_chemsys(chemical_system)
        elements = chemsys.split("-")

        # Get best structure for each formula in this system
        best_structures = self.get_best_structures_for_subsystem(chemsys)

        if not best_structures:
            logger.warning(f"No structures found for {chemsys}")
            return None, {}

        logger.info(
            f"Computing hull for {chemsys} with {len(best_structures)} formulas"
        )

        # Build phase diagram entries using ComputedEntry
        # (pymatgen's plotter shows labels as "Formula (entry_id)")
        entries = []
        structure_map: Dict[str, StoredStructure] = {}  # reduced_formula -> structure

        for formula, structure in best_structures.items():
            comp = Composition(formula)
            # Use energy per atom * num atoms in reduced formula
            energy = structure.energy_per_atom * comp.num_atoms
            # Use true formula + space group as entry_id for phase diagram labels
            # This shows the actual stoichiometry instead of reduced formula
            sg = structure.space_group_symbol or "?"
            sg_label = sg.replace("/", "-")
            entry = ComputedEntry(comp, energy, entry_id=f"{formula} ({sg_label})")
            entries.append(entry)
            # Key by reduced_formula to match lookup later
            structure_map[comp.reduced_formula] = structure

        if len(entries) < 2:
            logger.warning(f"Need at least 2 entries for hull, got {len(entries)}")
            return None, {}

        # Build phase diagram
        try:
            pd = PhaseDiagram(entries)
        except Exception as e:
            logger.error(f"Failed to build phase diagram for {chemsys}: {e}")
            return None, {}

        # Compute e_above_hull for each structure
        e_above_hull_map: Dict[str, float] = {}
        hull_structure_ids: List[str] = []

        for entry in entries:
            e_hull = pd.get_e_above_hull(entry)
            formula = entry.composition.reduced_formula
            e_above_hull_map[formula] = e_hull

            if e_hull < 1e-6:  # On hull
                hull_structure_ids.append(structure_map[formula].id)

        # Update database
        if update_database:
            now = datetime.now().isoformat()

            # Clear old hull entries for this chemsys to avoid duplicates
            # (e.g., if a different structure for the same formula was previously "best")
            self.conn.execute("DELETE FROM hull_entries WHERE chemsys = ?", (chemsys,))

            for formula, e_hull in e_above_hull_map.items():
                structure = structure_map[formula]
                is_on_hull = e_hull < 1e-6

                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO hull_entries
                    (chemsys, structure_id, e_above_hull, is_on_hull, computed_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (chemsys, structure.id, e_hull, 1 if is_on_hull else 0, now),
                )

            # Save hull snapshot
            snapshot_id = str(uuid.uuid4())
            self.conn.execute(
                """
                INSERT INTO hull_snapshots
                (id, chemsys, computed_at, num_entries, hull_entries, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    chemsys,
                    now,
                    len(entries),
                    json.dumps(hull_structure_ids),
                    json.dumps({"e_above_hull": e_above_hull_map}),
                ),
            )

            self._commit()
            logger.info(
                f"Updated hull for {chemsys}: {len(hull_structure_ids)} entries on hull"
            )

        return pd, e_above_hull_map

    def get_hull_entries(
        self, chemical_system: str, e_above_hull_cutoff: float = 0.0
    ) -> List[StoredStructure]:
        """
        Get structures on or near the convex hull.

        Args:
            chemical_system: Chemical system (e.g., "Fe-Mn-Co")
            e_above_hull_cutoff: Maximum e_above_hull in eV/atom (0 = only hull entries)

        Returns:
            List of StoredStructure objects on/near hull, sorted by e_above_hull
        """
        chemsys = self.normalize_chemsys(chemical_system)

        rows = self.conn.execute(
            """
            SELECT s.*, h.e_above_hull, h.is_on_hull
            FROM structures s
            JOIN hull_entries h ON s.id = h.structure_id
            WHERE h.chemsys = ? AND h.e_above_hull <= ?
            ORDER BY h.e_above_hull ASC
            """,
            (chemsys, e_above_hull_cutoff),
        ).fetchall()

        structures = []
        for row in rows:
            s = self._row_to_structure(row)
            s.e_above_hull = row["e_above_hull"]
            s.is_on_hull = bool(row["is_on_hull"])
            structures.append(s)

        return structures

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
        now = datetime.now().isoformat()

        self.conn.execute(
            """
            INSERT INTO runs (id, chemical_system, elements, parameters, started_at, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                chemsys,
                json.dumps(elements),
                json.dumps(parameters) if parameters else None,
                now,
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
        now = datetime.now().isoformat()

        self.conn.execute(
            """
            UPDATE runs SET
                completed_at = ?,
                num_candidates = ?,
                num_successful = ?,
                num_failed = ?
            WHERE id = ?
            """,
            (now, num_candidates, num_successful, num_failed, run_id),
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

    def get_run(self, run_id: str) -> Optional[ExplorationRun]:
        """Get a run by ID."""
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

        if row is None:
            return None

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

    def get_runs_for_system(self, chemical_system: str) -> List[ExplorationRun]:
        """Get all runs for a chemical system."""
        chemsys = self.normalize_chemsys(chemical_system)

        rows = self.conn.execute(
            "SELECT * FROM runs WHERE chemical_system = ? ORDER BY started_at DESC",
            (chemsys,),
        ).fetchall()

        return [
            ExplorationRun(
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
            for row in rows
        ]

    # -------------------- Statistics & Queries --------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}

        stats["total_structures"] = self.conn.execute(
            "SELECT COUNT(*) FROM structures"
        ).fetchone()[0]

        stats["valid_structures"] = self.conn.execute(
            "SELECT COUNT(*) FROM structures WHERE is_valid = 1"
        ).fetchone()[0]

        stats["total_runs"] = self.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[
            0
        ]

        stats["unique_formulas"] = self.conn.execute(
            "SELECT COUNT(DISTINCT formula) FROM structures WHERE is_valid = 1"
        ).fetchone()[0]

        stats["unique_chemsys"] = self.conn.execute(
            "SELECT COUNT(DISTINCT chemsys) FROM structures WHERE is_valid = 1"
        ).fetchone()[0]

        # Get list of explored systems
        rows = self.conn.execute(
            "SELECT DISTINCT chemical_system FROM runs ORDER BY chemical_system"
        ).fetchall()
        stats["explored_systems"] = [row[0] for row in rows]

        return stats

    def list_explored_systems(self) -> List[str]:
        """Get list of all explored chemical systems."""
        rows = self.conn.execute(
            "SELECT DISTINCT chemical_system FROM runs ORDER BY chemical_system"
        ).fetchall()
        return [row[0] for row in rows]

    def list_formulas(self, chemical_system: Optional[str] = None) -> List[str]:
        """Get list of all formulas, optionally filtered by chemical system."""
        if chemical_system:
            chemsys = self.normalize_chemsys(chemical_system)
            rows = self.conn.execute(
                """
                SELECT DISTINCT formula FROM structures 
                WHERE chemsys = ? AND is_valid = 1
                ORDER BY formula
                """,
                (chemsys,),
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
        limit: int = 100,
    ) -> List[StoredStructure]:
        """
        Search structures with various filters.

        Args:
            formula_pattern: SQL LIKE pattern for formula (e.g., "Fe%Mn%")
            elements: List of elements that must all be present
            max_energy_per_atom: Maximum energy per atom
            space_group: Required space group number
            limit: Maximum results to return

        Returns:
            List of matching StoredStructure objects
        """
        conditions = ["is_valid = 1"]
        params: List[Any] = []

        if formula_pattern:
            conditions.append("formula LIKE ?")
            params.append(formula_pattern)

        if max_energy_per_atom is not None:
            conditions.append("energy_per_atom <= ?")
            params.append(max_energy_per_atom)

        if space_group is not None:
            conditions.append("space_group_number = ?")
            params.append(space_group)

        query = f"""
            SELECT * FROM structures
            WHERE {' AND '.join(conditions)}
            ORDER BY energy_per_atom ASC
            LIMIT ?
        """
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        structures = [self._row_to_structure(row) for row in rows]

        # Filter by elements if specified
        if elements:
            required = set(elements)
            structures = [s for s in structures if required.issubset(set(s.elements))]

        return structures

    # -------------------- Migration --------------------

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
        import sqlite3 as sqlite

        run_dir = Path(run_directory)
        db_path = run_dir / "exploration.db"

        if not db_path.exists():
            raise ValueError(f"No database found at {db_path}")

        # Connect to the run's database
        run_conn = sqlite.connect(str(db_path))
        run_conn.row_factory = sqlite.Row

        # Get run metadata
        metadata = {}
        for row in run_conn.execute("SELECT key, value FROM run_metadata"):
            metadata[row["key"]] = row["value"]

        chemical_system = metadata.get("chemical_system", "unknown")
        started_at = metadata.get("started_at")
        completed_at = metadata.get("completed_at")

        # Create a run record
        run_id = self.create_run(
            chemical_system=chemical_system,
            parameters={
                "imported_from": str(run_dir),
                "original_metadata": metadata,
            },
            notes=f"Imported from {run_dir.name}",
        )

        # Update timestamps
        if started_at:
            self.conn.execute(
                "UPDATE runs SET started_at = ? WHERE id = ?",
                (started_at, run_id),
            )
        if completed_at:
            self.conn.execute(
                "UPDATE runs SET completed_at = ? WHERE id = ?",
                (completed_at, run_id),
            )
        self._commit()

        # Import structures
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

                # Try to load CIF content
                cif_content = None
                if cif_filename and structures_dir.exists():
                    cif_path = structures_dir / cif_filename
                    if cif_path.exists():
                        cif_content = cif_path.read_text()

                # Add to unified database
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

                # Track affected chemical systems for hull recomputation
                elements = sorted(stoichiometry.keys())
                chemsys = "-".join(elements)
                affected_systems.add(chemsys)

            except Exception as e:
                logger.warning(f"Failed to import {row['formula']}: {e}")
                skipped += 1
                continue

        run_conn.close()

        # Update run stats
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

        # Recompute hulls for affected systems
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

        # Find all run directories
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

                # Track affected systems
                # Extract chemsys from directory name
                name = run_dir.name
                if name.startswith("exploration_"):
                    parts = name.split("_")
                    if len(parts) >= 2:
                        chemsys = parts[1]
                        affected_systems.add(chemsys)

            except Exception as e:
                logger.error(f"Failed to import {run_dir}: {e}")
                results[str(run_dir)] = (0, 0)

        # Recompute all affected hulls at the end
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
