# GGen: Accelerating Materials Discovery

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**GGen** is a toolkit for discovering novel crystalline materials through computational exploration. Given a set of elements, GGen systematically generates crystal structures across all feasible stoichiometries, optimizes them using machine learning interatomic potentials (MLIPs), and constructs phase diagrams to identify thermodynamically stable candidates. For the most promising phases, phonon calculations verify dynamical stability—a key indicator of synthesizability.

The goal: **find materials that could actually exist in nature or be synthesized in a lab.**

## Installation

```bash
git clone https://github.com/ourofoundation/ggen.git
cd ggen
pip install -e .
```

## CLI Usage

GGen is designed for command-line exploration of chemical systems. The typical workflow:

1. **Explore** a chemical system to find low-energy structures
2. **Analyze** the phase diagram to identify stable candidates
3. **Validate** promising phases with phonon calculations

### Exploring a Chemical System

```bash
# Explore the Fe-Mn-Si ternary system
python scripts/explore_system.py Fe-Mn-Si

# Control search depth
python scripts/explore_system.py Fe-Mn-Si --max-atoms 24 --num-trials 25

# Focus on specific crystal systems
python scripts/explore_system.py Li-Co-O --crystal-systems hexagonal trigonal

# Faster exploration with parallel workers
python scripts/explore_system.py Fe-Sn-B -j 4
```

**What happens:** GGen enumerates all stoichiometries up to `--max-atoms` (default: 20), generates `--num-trials` candidate structures per stoichiometry (default: 15), and relaxes each using the ORB force field. Results are stored in a unified SQLite database that persists across runs—structures from Fe-Mn explored in one run are automatically reused when you explore Fe-Mn-Co later.

**Output:**
```
============================================================
EXPLORING: Fe-Mn-Si
============================================================
Database: ./ggen.db

Starting exploration...
  [████████████████████████████████████████] 87/87 stoichiometries

============================================================
RESULTS: Fe-Mn-Si
============================================================
Newly generated:           52
Reused from database:      35
Time elapsed:              127.3s

Stable/Near-Stable Phases (E_hull < 150 meV/atom)
------------------------------------------------------------
  Fe3Si        E=-7.2341 eV/atom  SG=Fm-3m       E_hull=  0.0 meV  dyn:✓  [db]
  Mn3Si        E=-6.8924 eV/atom  SG=Fm-3m       E_hull=  0.0 meV  dyn:?  [new]
  FeMnSi       E=-6.5112 eV/atom  SG=P6_3/mmc    E_hull= 12.3 meV  dyn:?  [new]
  ...
```

### CLI Options Reference

```bash
python scripts/explore_system.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--max-atoms` | 20 | Maximum atoms per unit cell |
| `--min-atoms` | 2 | Minimum atoms per unit cell |
| `--num-trials` | 15 | Generation attempts per stoichiometry |
| `--max-stoichiometries` | 100 | Cap on stoichiometries to explore |
| `--crystal-systems` | all | Filter: `cubic`, `hexagonal`, `tetragonal`, etc. |
| `-j`, `--workers` | 1 | Parallel workers for generation |
| `--e-above-hull` | 0.15 | Energy cutoff (eV) for "stable" phases |
| `--compute-phonons` | off | Run phonon calculations during exploration |
| `--hide-unstable` | off | Hide dynamically unstable phases from output |
| `--skip-existing` | off | Skip formulas that already exist in database |
| `--db-path` | ./ggen.db | Path to unified structure database |
| `--output-dir` | ./runs | Directory for CIF files and plots |
| `--seed` | random | Random seed for reproducibility |

### Phonon Stability (Dynamical Stability)

A structure can be thermodynamically stable (on the convex hull) but dynamically unstable—meaning it would spontaneously distort into a different structure. Phonon calculations detect imaginary vibrational modes that indicate this instability.

Since phonon calculations can be expensive (~10s per structure), GGen separates them from the initial exploration:

```bash
# First, explore quickly (no phonons)
python scripts/explore_system.py Zn-Cu-Sn

# Then compute phonons for promising candidates
python scripts/backfill_phonons.py --e-above-hull 0.1

# Target a specific system
python scripts/backfill_phonons.py --system Zn-Cu-Sn --e-above-hull 0.05

# Preview what would be computed
python scripts/backfill_phonons.py --dry-run

# Limit the batch size
python scripts/backfill_phonons.py --max-structures 20
```

If you want phonons computed during exploration (slower but all-in-one):

```bash
python scripts/explore_system.py Zn-Cu-Sn --compute-phonons
```

**Interpreting results:**
- `dyn:✓` — Dynamically stable (no imaginary modes) → likely synthesizable
- `dyn:✗` — Dynamically unstable (imaginary modes present) → would distort
- `dyn:?` — Not yet tested

### Running Multiple Systems

Use GNU parallel to explore multiple systems concurrently:

```bash
parallel python scripts/explore_system.py ::: Fe-Mn-Si Li-Co-O Zn-Sn-Cu Na-P-S
```

Each run shares the unified database, so common subsystems (e.g., Fe-Mn) are explored once and reused.

## Key Concepts

### Phase Diagrams & Convex Hulls

A **phase diagram** shows which compositions are thermodynamically stable for a given set of elements. The **convex hull** is the lower envelope of energies—structures on the hull are stable against decomposition into competing phases. Structures above the hull are metastable: the distance above hull (E_hull) indicates how much energy would be released if they decomposed.

GGen computes phase diagrams automatically using [pymatgen](https://pymatgen.org/) and generates interactive HTML plots (via Plotly) showing all explored structures.

### Thermodynamic vs. Dynamical Stability

- **Thermodynamic stability** (energy above hull): Can this structure exist without decomposing?
- **Dynamical stability** (phonon modes): Is this structure mechanically stable, or would it spontaneously distort?

A promising synthesis candidate should be both thermodynamically stable (or nearly so) and dynamically stable. GGen lets you filter for exactly this.

## Features

- **Crystal Generation**: Generate structures from chemical formulas via PyXtal with automatic space group selection
- **MLIP Optimization**: Geometry relaxation using ORB force field models (fast and accurate)
- **Phase Diagram Construction**: Convex hull analysis with interactive Plotly visualizations
- **Phonon Calculations**: Detect imaginary modes to assess dynamical stability
- **Unified Database**: SQLite storage with cross-system structure sharing
- **Incremental Exploration**: Skip already-explored formulas; build on previous runs
- **Structure Mutations**: Lattice scaling, shearing, substitution, site operations for evolutionary optimization
- **Multiple Export Formats**: CIF, XYZ, JSON, ASE trajectory files
- **Trajectory Tracking**: Record structure evolution through mutations and optimizations

## Python API

For programmatic access, GGen provides two main classes:

### `ChemistryExplorer` — Systematic Exploration

```python
from ggen import ChemistryExplorer

explorer = ChemistryExplorer(output_dir="./runs")

result = explorer.explore(
    chemical_system="Fe-Mn-Si",
    max_atoms=16,
    num_trials=15,
    optimize=True,
    preserve_symmetry=True
)

print(f"Found {len(result.hull_entries)} phases on the convex hull")

# Get stable candidates
stable = explorer.get_stable_candidates(result, e_above_hull_cutoff=0.025)
for s in stable:
    print(f"{s.formula}: {s.energy_per_atom:.4f} eV/atom, E_hull={s.e_above_hull*1000:.1f} meV")

# Generate interactive phase diagram
fig = explorer.plot_phase_diagram(result)
fig.write_html("phase_diagram.html")
```

### `GGen` — Single Structure Operations

```python
from ggen import GGen

ggen = GGen()

# Generate a crystal
result = ggen.generate_crystal("BaTiO3", num_trials=10, optimize_geometry=True)
print(f"Space group: {result['final_space_group_symbol']}")
print(f"Energy: {result['best_crystal_energy']:.4f} eV")

# Apply mutations
ggen.scale_lattice(1.05)
ggen.substitute("Ba", "Sr", fraction=0.5)
ggen.jitter_sites(sigma=0.01)

# Export trajectory
ggen.export_trajectory("evolution.traj")
```

## Output Structure

Each exploration run creates a timestamped directory:

```
runs/exploration_Fe-Mn-Si_20260105_143500/
├── structures/           # CIF files for all generated structures
│   ├── Fe2Mn_Im-3m.cif
│   ├── FeMnSi_P63-mmc.cif
│   └── ...
├── phase_diagram.html    # Interactive phase diagram
├── phase_diagram.png     # Static image (if kaleido installed)
└── summary.json          # JSON summary of results
```

The unified database (`ggen.db` by default) stores all structures across all runs.

## API Reference

<details>
<summary><strong>GGen Class</strong></summary>

**Initialization:**
```python
GGen(
    calculator=None,        # ASE calculator (default: ORB)
    random_seed=None,       # For reproducibility
    enable_trajectory=True  # Track structure evolution
)
```

**Key Methods:**
- `generate_crystal(formula, space_group=None, num_trials=10, optimize_geometry=False)`
- `set_structure(structure, add_to_trajectory=True)`
- `get_structure()`
- `optimize_geometry(max_steps=400, fmax=0.01, relax_cell=True)`

**Mutation Methods:**
- `scale_lattice(scale_factor, isotropic=True)`
- `shear_lattice(angle_deltas)`
- `substitute(element_from, element_to, fraction=1.0)`
- `add_site(element, coordinates, coords_are_cartesian=False)`
- `remove_site(site_indices=None, element=None, max_remove=None)`
- `move_site(site_index, displacement=None, new_coordinates=None)`
- `jitter_sites(sigma=0.01, element=None)`
- `symmetry_break(displacement_scale=0.01, angle_perturbation=0.1)`
- `change_space_group(target_space_group, symprec=0.1)`

**Export Methods:**
- `export_trajectory(filename)` — Export to CIF, XYZ, JSON, or ASE `.traj` (auto-detected from extension)

</details>

<details>
<summary><strong>ChemistryExplorer Class</strong></summary>

**Initialization:**
```python
ChemistryExplorer(
    calculator=None,     # ASE calculator (default: ORB, lazy-loaded)
    random_seed=None,    # For reproducibility
    output_dir=None      # Base directory for results
)
```

**Main Methods:**
- `explore(chemical_system, ...)` — Run full exploration
- `get_stable_candidates(result, e_above_hull_cutoff=0.025)` — Filter stable phases
- `plot_phase_diagram(result, show_unstable=0.1)` — Generate interactive plot
- `export_summary(result, output_path=None)` — Export JSON summary

**Utility Methods:**
- `parse_chemical_system(system)` — Parse "Li-Co-O" → ["Co", "Li", "O"]
- `enumerate_stoichiometries(elements, max_atoms=12, ...)`
- `find_previous_runs(chemical_system)`
- `load_structures_from_previous_runs(chemical_system)`
- `load_run(run_directory)`

</details>

<details>
<summary><strong>Data Classes</strong></summary>

**CandidateResult** — A generated structure candidate:
- `formula`, `stoichiometry`, `energy_per_atom`, `total_energy`
- `num_atoms`, `space_group_number`, `space_group_symbol`
- `structure` (pymatgen), `cif_path`, `generation_metadata`
- `is_valid`, `error_message`
- `is_dynamically_stable`, `phonon_result`

**ExplorationResult** — Complete exploration results:
- `chemical_system`, `elements`
- `num_candidates`, `num_successful`, `num_failed`
- `candidates`, `phase_diagram`, `hull_entries`
- `run_directory`, `database_path`, `total_time_seconds`

</details>

## Dependencies

**Core:** numpy, scipy, pymatgen, pyxtal, ase, torch  
**ML Potentials:** orb-models, torch-sim-atomistic  
**Phonons:** phonopy, seekpath  
**Visualization:** plotly, matplotlib  
**Utilities:** tqdm, requests, pynanoflann

## Contributing

Contributions welcome! Please open an issue to discuss major changes before submitting a PR.

## License

MIT License — see [LICENSE](LICENSE)

## Citation

```bibtex
@software{ggen2026,
  title={GGen: Crystal Generation and Mutation Library},
  author={Matt Moderwell},
  year={2026},
  url={https://github.com/ourofoundation/ggen}
}
```

## Acknowledgments

Built on [PyXtal](https://github.com/qzhu2017/PyXtal), [pymatgen](https://github.com/materialsproject/pymatgen), [ASE](https://wiki.fysik.dtu.dk/ase/), [orb-models](https://github.com/orbital-materials/orb-models), [phonopy](https://github.com/phonopy/phonopy), and [torch-sim](https://github.com/torchsim/torch-sim).
