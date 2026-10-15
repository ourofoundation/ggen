"""Reference data sources: published structures with their PBE energies.

Each source yields ``SourceRecord``s: a structure as published, with the raw
(uncorrected) PBE energy computed on it and, when the source publishes them,
the run settings and its own MP2020 correction. ``scripts/import.py`` writes
records into the database as source geometries and ``pbe`` calculations, so
adding a dataset means adding one iterator here.

Licenses travel with the records so outputs can attribute them.
"""

from __future__ import annotations

import bz2
import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from pymatgen.core import Structure

logger = logging.getLogger(__name__)

LICENSES = {
    "mp": "CC-BY-4.0",
    "alexandria": "CC-BY-4.0",
}


@dataclass
class SourceRecord:
    source: str
    source_id: str
    structure: Structure
    # Raw PBE (or PBE+U) energy per atom on ``structure``'s cell.
    energy_per_atom: Optional[float]
    # The source's own MP2020 correction, when it applies MP2020 exactly as
    # we would (Materials Project). Otherwise computed on import.
    correction_per_atom: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    space_group_number: Optional[int] = None
    space_group_symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# -------------------- Materials Project --------------------


def _mp_chemsys_batches(mpr, batch_size: int, elements: Optional[Sequence[str]]):
    """All chemical systems with materials, in batches (thermo has no element filter)."""
    kwargs: Dict[str, Any] = {"fields": ["material_id", "chemsys"]}
    if elements:
        kwargs["elements"] = list(elements)
    docs = mpr.materials.summary.search(**kwargs)
    systems = sorted({doc.chemsys for doc in docs})
    logger.info(f"Materials Project: {len(docs):,} materials in {len(systems):,} systems")
    for start in range(0, len(systems), batch_size):
        yield systems[start : start + batch_size]


def materials_project(
    api_key: str,
    elements: Optional[Sequence[str]] = None,
    batch_size: int = 250,
) -> Iterator[SourceRecord]:
    """GGA/GGA+U entries from Materials Project's ``GGA_GGA+U`` thermo documents.

    Materials only calculated with r2SCAN have no such entry and are skipped.
    The energy and correction are MP's own (uncorrected energy plus the
    MP2020 adjustment MP applied), on the GGA structure itself.
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        for systems in _mp_chemsys_batches(mpr, batch_size, elements):
            docs = mpr.materials.thermo.search(
                thermo_types=["GGA_GGA+U"],
                chemsys=systems,
                fields=[
                    "material_id",
                    "entries",
                    "symmetry",
                    "energy_per_atom",
                    "energy_above_hull",
                ],
            )
            for doc in docs:
                # The entry this thermo document's energy comes from.
                entries = [e for e in (doc.entries or {}).values() if e is not None]
                if not entries or doc.energy_per_atom is None:
                    continue
                entry = min(entries, key=lambda e: abs(e.energy_per_atom - doc.energy_per_atom))
                if entry.structure is None:
                    continue
                symmetry = doc.symmetry
                yield SourceRecord(
                    source="mp",
                    source_id=str(doc.material_id),
                    structure=entry.structure,
                    energy_per_atom=entry.uncorrected_energy_per_atom,
                    correction_per_atom=entry.correction_per_atom,
                    parameters={
                        "run_type": entry.parameters.get("run_type"),
                        "is_hubbard": entry.parameters.get("is_hubbard"),
                        "hubbards": entry.parameters.get("hubbards") or {},
                    },
                    space_group_number=getattr(symmetry, "number", None),
                    space_group_symbol=getattr(symmetry, "symbol", None),
                    metadata={
                        "mp_entry_id": str(entry.entry_id),
                        "mp_energy_above_hull": doc.energy_above_hull,
                    },
                )


# -------------------- Alexandria --------------------


def _alexandria_record(entry: Dict[str, Any]) -> Optional[SourceRecord]:
    data = entry.get("data") or {}
    material_id = data.get("mat_id") or entry.get("entry_id")
    structure_dict = entry.get("structure")
    if not material_id or not structure_dict:
        return None
    try:
        structure = Structure.from_dict(structure_dict)
    except Exception:
        return None
    energy_total = data.get("energy_total")
    sg_number = data.get("spg")
    return SourceRecord(
        source="alexandria",
        source_id=str(material_id),
        structure=structure,
        # Alexandria's energy_total is the raw energy of this cell; its
        # energy_corrected follows a different scheme from MP's, so the
        # correction is recomputed on import.
        energy_per_atom=None if energy_total is None else energy_total / len(structure),
        space_group_number=int(sg_number) if sg_number else None,
        metadata={
            "alexandria_band_gap": data.get("band_gap_ind", data.get("band_gap_dir")),
            # PBE total magnetic moment of the cell (μB)
            "alexandria_total_mag": data.get("total_mag"),
            "alexandria_last_modified": data.get("last_modified"),
        },
    )


def alexandria(
    directory: Path,
    pattern: str = "alexandria_*.json.bz2",
    elements: Optional[Sequence[str]] = None,
) -> Iterator[SourceRecord]:
    """Alexandria PBE entries from the downloaded ``.json.bz2`` chunk files."""
    chunks = sorted(Path(directory).glob(pattern))
    if not chunks:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    wanted = set(elements or [])
    logger.info(f"Alexandria: {len(chunks)} chunk files in {directory}")
    for chunk in chunks:
        logger.info(f"  {chunk.name}")
        with bz2.open(chunk, "rt") as handle:
            payload = json.load(handle)
        for entry in payload.get("entries", []):
            if wanted:
                entry_elements = set((entry.get("data") or {}).get("elements") or [])
                if not wanted <= entry_elements:
                    continue
            record = _alexandria_record(entry)
            if record is not None:
                yield record
        del payload
        gc.collect()


def chunk_records(records: Iterator[SourceRecord], size: int) -> Iterator[List[SourceRecord]]:
    batch: List[SourceRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
