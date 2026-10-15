"""Energy surfaces structures can be compared on.

A surface is one consistent way of assigning energies: DFT (PBE/PBE+U from
the reference databases) or one machine-learned potential. Every energy in
the database belongs to exactly one surface, and a hull is only ever built
from energies of a single surface, so adding a model is one entry here plus
filling its energies.

Geometries are stored per surface too. ``SOURCE_GEOMETRY`` is the structure
as the source published it (PBE-relaxed for Materials Project and
Alexandria), which is what the ``pbe`` surface's energies refer to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

SOURCE_GEOMETRY = "source"


@dataclass(frozen=True)
class Surface:
    id: str
    kind: str  # "dft" or "mlip"
    label: str
    correction_scheme: str = "mp2020"
    # Model checkpoint for an MLIP surface, as ggen.calculator names it.
    model: Optional[str] = None

    @property
    def geometry_kind(self) -> str:
        """The geometry slot this surface's energies were computed on."""
        return SOURCE_GEOMETRY if self.kind == "dft" else self.id


PBE = Surface(id="pbe", kind="dft", label="PBE (MP2020-corrected)")
ORB_V3 = Surface(
    id="orb-v3-conservative-inf-mpa",
    kind="mlip",
    label="Orb v3 conservative inf MPA",
    model="orb_v3_conservative_inf_mpa",
)

SURFACES: Dict[str, Surface] = {s.id: s for s in (PBE, ORB_V3)}

# The surface GGen generates and relaxes on.
DEFAULT_SURFACE = ORB_V3.id


def get_surface(surface_id: str) -> Surface:
    try:
        return SURFACES[surface_id]
    except KeyError:
        raise ValueError(
            f"Unknown energy surface {surface_id!r}; known: {', '.join(SURFACES)}"
        ) from None
