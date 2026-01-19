"""
Modal script for running ggen with GPU acceleration.

Usage:
    # Run exploration from local CLI
    modal run scripts/modal.py --system Fe-Co-Bi --max-atoms 16

    # Deploy for remote invocation
    modal deploy scripts/modal.py

Example local invocation after deploy:
    python -c "import modal; f = modal.Function.from_name('ggen', 'explore'); print(f.remote('Fe-Co-Bi', max_atoms=16))"
"""

import modal
from pathlib import Path

# Get the ggen project root (parent of scripts/)
GGEN_ROOT = Path(__file__).parent.parent

# Volume for persisting the database
volume = modal.Volume.from_name("ggen", create_if_missing=True, version=2)

# Mount local ggen package into the container
ggen_mount = modal.Mount.from_local_dir(
    GGEN_ROOT / "ggen",
    remote_path="/root/ggen_pkg/ggen",
)

# Image with all dependencies for GPU-accelerated ggen
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        # Core scientific stack
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        # Materials science libraries
        "pymatgen>=2023.0.0",
        "pyxtal>=0.5.0",
        "ase>=3.22.0",
        # ML potentials (GPU accelerated)
        "orb-models>=0.5.5",
        "pynanoflann",
        "torch-sim-atomistic>=0.5.0",
        "cuml-cu12==25.2.*",
        # Phonon calculations
        "phonopy>=2.20.0",
        "seekpath>=2.0",
        # Utilities
        "tqdm>=4.60.0",
        "requests>=2.25.0",
        "matplotlib>=3.0",
    )
)

app = modal.App(
    name="ggen",
    image=image,
    volumes={"/data": volume},
)


@app.function(
    gpu="A10G",
    timeout=3600,  # 1 hour timeout
    volumes={"/data": volume},
    mounts=[ggen_mount],
)
def explore(
    system: str,
    max_atoms: int = 20,
    min_atoms: int = 2,
    num_trials: int = 15,
    max_stoichiometries: int = 100,
    skip_existing: bool = False,
    preserve_symmetry: bool = False,
    max_steps: int = 400,
    e_above_hull: float = 0.15,
    compute_phonons: bool = False,
    require_all_elements: bool = False,
    crystal_systems: list[str] | None = None,
    space_group: int | str | None = None,
    workers: int = 1,
) -> dict:
    """
    Explore a chemical system and generate phase diagrams with GPU acceleration.

    Args:
        system: Chemical system to explore, e.g. 'Fe-Mn-Si' or 'Li-Co-O'
        max_atoms: Maximum atoms per unit cell
        min_atoms: Minimum atoms per unit cell
        num_trials: Number of trials per stoichiometry
        max_stoichiometries: Maximum stoichiometries to explore
        skip_existing: Skip formulas that already exist in database
        preserve_symmetry: Preserve symmetry during optimization
        max_steps: Maximum optimization steps per structure
        e_above_hull: Energy above hull cutoff in eV/atom
        compute_phonons: Compute phonon stability (expensive)
        require_all_elements: Only generate formulas containing all elements
        crystal_systems: List of crystal systems to explore (e.g., ['cubic', 'tetragonal'])
        space_group: Specific space group to target (number or symbol)
        workers: Number of parallel workers for structure generation

    Returns:
        Dictionary with exploration results and statistics
    """
    import logging
    import sys
    import warnings

    # Add mounted ggen package to path
    sys.path.insert(0, "/root/ggen_pkg")

    from ggen import ChemistryExplorer, StructureDatabase

    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("ggen.modal")

    # Database path on the persistent volume
    db_path = "/data/ggen.db"
    output_dir = "/data/runs"

    logger.info(f"Starting GPU-accelerated exploration of {system}")
    logger.info(f"Database: {db_path}")

    # Check GPU availability
    import torch

    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected, running on CPU")

    # Initialize database
    db = StructureDatabase(db_path)
    chemsys = db.normalize_chemsys(system)

    # Get stats before exploration
    global_stats = db.get_statistics()
    logger.info(f"Database has {global_stats['total_structures']} structures")

    # Initialize explorer
    explorer = ChemistryExplorer(
        output_dir=output_dir,
        database=db,
    )

    # Parse space group if it's a symbol
    sg = None
    if space_group:
        try:
            sg = int(space_group)
        except ValueError:
            from pyxtal.symmetry import Group

            g = Group(space_group, dim=3)
            sg = g.number

    # Run exploration
    result = explorer.explore(
        chemical_system=system,
        max_atoms=max_atoms,
        min_atoms=min_atoms,
        num_trials=num_trials,
        optimize=True,
        include_binaries=True,
        include_ternaries=True,
        require_all_elements=require_all_elements,
        max_stoichiometries=max_stoichiometries,
        crystal_systems=crystal_systems,
        space_group=sg,
        skip_existing_formulas=skip_existing,
        preserve_symmetry=preserve_symmetry,
        num_workers=workers,
        show_progress=True,
        keep_structures_in_memory=False,
        use_unified_database=True,
        compute_phonons=compute_phonons,
        optimization_max_steps=max_steps,
    )

    # Commit volume changes
    volume.commit()

    # Get final stats
    global_stats_after = db.get_statistics()

    # Get stable phases
    stable_phases = db.get_hull_entries(chemsys, e_above_hull_cutoff=e_above_hull)

    # Build results
    results = {
        "chemical_system": result.chemical_system,
        "num_candidates": result.num_candidates,
        "num_failed": result.num_failed,
        "total_time_seconds": result.total_time_seconds,
        "database_stats": {
            "total_structures": global_stats_after["total_structures"],
            "unique_formulas": global_stats_after["unique_formulas"],
            "unique_chemsys": global_stats_after["unique_chemsys"],
        },
        "stable_phases": [
            {
                "formula": s.formula,
                "energy_per_atom": s.energy_per_atom,
                "space_group": s.space_group_symbol,
                "e_above_hull": s.e_above_hull,
                "is_dynamically_stable": s.is_dynamically_stable,
            }
            for s in stable_phases
        ],
    }

    db.close()

    logger.info(f"Exploration complete! Found {len(stable_phases)} stable phases")

    return results


@app.function(
    gpu="A10G",
    timeout=7200,  # 2 hour timeout for batch runs
    volumes={"/data": volume},
    mounts=[ggen_mount],
)
def explore_batch(systems: list[str], **kwargs) -> list[dict]:
    """
    Explore multiple chemical systems sequentially.

    Args:
        systems: List of chemical systems to explore
        **kwargs: Arguments passed to explore()

    Returns:
        List of results for each system
    """
    results = []
    for system in systems:
        result = explore.local(system, **kwargs)
        results.append(result)

    return results


@app.function(volumes={"/data": volume})
def upload_db(db_bytes: bytes):
    """
    Upload local database to Modal volume.
    Called internally by upload_local_db().
    """
    with open("/data/ggen.db", "wb") as f:
        f.write(db_bytes)
    volume.commit()
    print(f"Uploaded database ({len(db_bytes) / 1024 / 1024:.2f} MB)")


@app.function(volumes={"/data": volume})
def download_db() -> bytes:
    """
    Download database from Modal volume.
    Called internally by download_remote_db().
    """
    import os

    db_path = "/data/ggen.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError("No database found on volume")

    with open(db_path, "rb") as f:
        return f.read()


@app.function(volumes={"/data": volume}, mounts=[ggen_mount])
def get_db_stats() -> dict:
    """Get statistics from the remote database."""
    import sys

    sys.path.insert(0, "/root/ggen_pkg")

    from ggen import StructureDatabase

    db = StructureDatabase("/data/ggen.db")
    stats = db.get_statistics()
    db.close()
    return stats


@app.local_entrypoint()
def main(
    system: str,
    max_atoms: int = 20,
    min_atoms: int = 2,
    num_trials: int = 15,
    max_stoichiometries: int = 100,
    skip_existing: bool = False,
    preserve_symmetry: bool = False,
    compute_phonons: bool = False,
):
    """
    Local entrypoint for running exploration via Modal CLI.

    Usage:
        modal run scripts/modal.py --system Fe-Co-Bi --max-atoms 16
    """
    print(f"🚀 Starting remote GPU exploration of {system}")

    result = explore.remote(
        system=system,
        max_atoms=max_atoms,
        min_atoms=min_atoms,
        num_trials=num_trials,
        max_stoichiometries=max_stoichiometries,
        skip_existing=skip_existing,
        preserve_symmetry=preserve_symmetry,
        compute_phonons=compute_phonons,
    )

    print("\n" + "=" * 60)
    print(f"✅ Exploration complete: {result['chemical_system']}")
    print("=" * 60)
    print(f"  Candidates explored: {result['num_candidates']}")
    print(f"  Failed: {result['num_failed']}")
    print(f"  Time: {result['total_time_seconds']:.1f}s")
    print(f"\n  Database totals:")
    print(f"    Structures: {result['database_stats']['total_structures']}")
    print(f"    Formulas: {result['database_stats']['unique_formulas']}")
    print(f"    Systems: {result['database_stats']['unique_chemsys']}")

    if result["stable_phases"]:
        print(f"\n  Stable phases ({len(result['stable_phases'])}):")
        for p in result["stable_phases"][:10]:  # Show top 10
            dyn = (
                "✓"
                if p["is_dynamically_stable"]
                else "✗" if p["is_dynamically_stable"] is False else "?"
            )
            e_hull = p["e_above_hull"] or 0
            print(
                f"    {p['formula']:12s}  E={p['energy_per_atom']:.4f}  "
                f"SG={p['space_group']:10s}  E_hull={e_hull*1000:5.1f}meV  dyn:{dyn}"
            )
        if len(result["stable_phases"]) > 10:
            print(f"    ... and {len(result['stable_phases']) - 10} more")

    return result


# ============================================================================
# Helper functions for local use
# ============================================================================


def upload_local_db(local_db_path: str | Path = None):
    """
    Upload local database to Modal volume.

    Usage:
        from scripts.modal import upload_local_db
        upload_local_db()  # Uses default ./ggen.db
        upload_local_db("/path/to/ggen.db")
    """
    if local_db_path is None:
        local_db_path = GGEN_ROOT / "ggen.db"
    else:
        local_db_path = Path(local_db_path)

    if not local_db_path.exists():
        raise FileNotFoundError(f"Database not found: {local_db_path}")

    print(f"📤 Uploading {local_db_path} to Modal volume...")
    with open(local_db_path, "rb") as f:
        db_bytes = f.read()

    with app.run():
        upload_db.remote(db_bytes)

    print("✅ Upload complete!")


def download_remote_db(local_db_path: str | Path = None):
    """
    Download database from Modal volume to local path.

    Usage:
        from scripts.modal import download_remote_db
        download_remote_db()  # Saves to ./ggen.db
        download_remote_db("/path/to/save/ggen.db")
    """
    if local_db_path is None:
        local_db_path = GGEN_ROOT / "ggen.db"
    else:
        local_db_path = Path(local_db_path)

    print(f"📥 Downloading database from Modal volume...")

    with app.run():
        db_bytes = download_db.remote()

    with open(local_db_path, "wb") as f:
        f.write(db_bytes)

    print(f"✅ Downloaded to {local_db_path} ({len(db_bytes) / 1024 / 1024:.2f} MB)")


def run_explore(system: str, **kwargs) -> dict:
    """
    Run exploration remotely with GPU acceleration.

    Usage:
        from scripts.modal import run_explore
        result = run_explore("Fe-Co-Bi", max_atoms=16)

    Args:
        system: Chemical system to explore
        **kwargs: Additional arguments (see explore() for options)

    Returns:
        Dictionary with exploration results
    """
    with app.run():
        return explore.remote(system, **kwargs)
