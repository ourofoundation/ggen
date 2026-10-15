"""The structure database shared across Modal apps on one Volume.

A Modal Volume keeps the last commit of a file, so two containers writing
the SQLite file would silently drop each other's rows. Only the ggen app's
``GGenWriter`` (one container at most) writes it. Everyone else:

- reads a snapshot: ``reader()`` reloads the Volume (at most every
  ``reload_interval`` seconds, and only when nothing in the container has
  the file open) and opens it read-only;
- or, to write, works on a local copy: ``working_copy(systems)`` extracts
  the systems to local disk, yields a writable database, then exports what
  changed and hands that shard to the writer (``ingest_remote``).

Usage, in any app that mounts the Volume::

    shared = SharedDatabase(volume, "/materials/data/ggen.v2.db")
    with shared.reader() as db:
        ...
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Union

from .database import StructureDatabase
from .surfaces import DEFAULT_SURFACE

logger = logging.getLogger(__name__)

WRITER_APP = "ggen"
WRITER_CLASS = "GGenWriter"
# Deployments other than production (a dev materials-compute, say) set this
# to their own ggen app, or to "none" to keep their writes out of the
# shared database.
WRITER_APP_ENV = "GGEN_WRITER_APP"


def ingest_remote(shard: bytes, recompute_hulls: bool = False) -> dict:
    """Queue a shard on the deployed writer without waiting for the merge.

    The writer takes one call at a time and may be busy with a long
    maintenance task, so callers do not block on it; the shard is merged
    when the writer gets to it.
    """
    import os

    import modal

    app_name = os.getenv(WRITER_APP_ENV, WRITER_APP)
    if app_name == "none":
        logger.info("%s=none: shard of %d bytes not sent", WRITER_APP_ENV, len(shard))
        return {"queued": None}
    writer = modal.Cls.from_name(app_name, WRITER_CLASS)()
    call = writer.ingest.spawn(shard, recompute_hulls=recompute_hulls)
    return {"queued": call.object_id}


class SharedDatabase:
    def __init__(
        self,
        volume: Any,
        path: Union[str, Path],
        reload_interval: float = 30.0,
    ):
        self.volume = volume
        self.path = Path(path)
        self.reload_interval = reload_interval
        self._lock = threading.Lock()
        # Weak references, so a reader a request forgot to close stops
        # blocking reloads once it is garbage.
        self._readers: "weakref.WeakSet[StructureDatabase]" = weakref.WeakSet()
        self._last_reload = 0.0

    def _maybe_reload(self) -> None:
        if time.time() - self._last_reload < self.reload_interval:
            return
        if any(reader._conn is not None for reader in list(self._readers)):
            return
        try:
            self.volume.reload()
            self._last_reload = time.time()
        except Exception as exc:
            # Files open elsewhere in the container block a reload; the
            # current view is then used until the next request.
            logger.info("Volume reload skipped: %s", exc)

    def open_reader(self, surface: Optional[str] = None) -> StructureDatabase:
        """A read-only snapshot. Close it when done."""
        with self._lock:
            self._maybe_reload()
            db = StructureDatabase(
                self.path,
                surface=surface or DEFAULT_SURFACE,
                readonly=True,
                snapshot=True,
            )
            db.conn  # open now, so a concurrent reload sees it
            self._readers.add(db)
        return db

    @contextmanager
    def reader(self, surface: Optional[str] = None) -> Iterator[StructureDatabase]:
        db = self.open_reader(surface)
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def working_copy(
        self,
        chemical_systems: Iterable[str],
        ingest: Callable[[bytes], dict] = ingest_remote,
        surface: Optional[str] = None,
    ) -> Iterator[StructureDatabase]:
        """A writable local copy of some systems; changes go to the writer on exit.

        The copy holds every structure in the systems and their subsystems,
        so hulls built from it match the shared database. Changes are sent
        even when the block raises, so work done before a failure is kept.
        """
        surface = surface or DEFAULT_SURFACE
        systems = list(chemical_systems)
        with tempfile.TemporaryDirectory(prefix="ggen-work-") as tmp:
            work_path = Path(tmp) / "work.db"
            with self.reader(surface) as reader:
                count = reader.extract_systems(work_path, systems)
            logger.info("Working copy of %s: %d structures", ", ".join(systems), count)

            since = datetime.now().isoformat()
            db = StructureDatabase(work_path, surface=surface)
            try:
                yield db
            finally:
                delta_path = Path(tmp) / "delta.db"
                changed = db.export_changes(delta_path, since)
                db.close()
                if changed:
                    result = ingest(delta_path.read_bytes())
                    logger.info("Sent %d changed structures to the writer: %s", changed, result)

    @contextmanager
    def exclusive_local_copy(self) -> Iterator[Path]:
        """The whole file on local disk, written back and committed on success.

        Every write by the writer goes through this: SQLite is much faster on
        local disk, the Volume only ever sees a complete file (swapped in by
        rename), and since the writer is the only writer nothing changes the
        file in the meantime.
        """
        import shutil

        with tempfile.TemporaryDirectory(prefix="ggen-maintenance-") as tmp:
            local = Path(tmp) / self.path.name
            self.volume.reload()
            shutil.copyfile(self.path, local)
            yield local
            db = StructureDatabase(local)
            db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            db.close()
            staged = self.path.with_suffix(".staged")
            shutil.copyfile(local, staged)
            staged.replace(self.path)
            # The writer checkpoints after every merge, so these are empty;
            # removed so they can never be replayed against the new file.
            for suffix in ("-wal", "-shm"):
                Path(f"{self.path}{suffix}").unlink(missing_ok=True)
            self.volume.commit()

    def merge_shard(self, shard: bytes, recompute_hulls: bool = False) -> dict:
        """Merge a shard and commit the Volume. Only the writer calls this.

        The merge runs on a local copy that replaces the file in one rename,
        because a Volume can commit in the background at any moment and must
        never capture the SQLite file half-written.

        Stored hull distances are not refreshed by default: routes compute
        hulls fresh, and the writer's ``hulls`` maintenance task refreshes
        them in bulk.
        """
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="ggen-shard-") as tmp:
            shard_path = Path(tmp) / "shard.db"
            shard_path.write_bytes(shard)
            with self.exclusive_local_copy() as local:
                db = StructureDatabase(local)
                try:
                    new, merged = db.merge_from(shard_path, recompute_hulls=recompute_hulls)
                finally:
                    db.close()
        return {
            "new_structures": new,
            "merged_into_existing": merged,
            "seconds": round(time.time() - started, 1),
        }
