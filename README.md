# GGen: Crystal Generation, Mutation & Phase Space Exploration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library for crystal structure generation, mutation, evolutionary optimization, and chemical space exploration. Built on top of PyXtal, pymatgen, and ASE, GGen provides an intuitive interface for generating, modifying, and analyzing crystal structures with built-in energy evaluation using ORB force field models.

## Features

### Core Crystal Operations
- **Crystal Generation**: Generate crystal structures from chemical formulas using PyXtal
- **Space Group Analysis**: Automatic space group compatibility checking and selection
- **Structure Mutations**: Comprehensive set of mutation operations for evolutionary optimization
- **Energy Evaluation**: Built-in energy calculation using ORB force field models
- **Trajectory Tracking**: Track structure evolution through mutations and optimizations
- **Multiple Export Formats**: Export structures as CIF, XYZ, JSON, or standard ASE `.traj` files
- **Structure Validation**: Built-in validation and repair mechanisms

### Chemical Space Exploration
- **Systematic Exploration**: Explore chemical systems (e.g., "Fe-Mn-Si") across all stoichiometries
- **Phase Diagram Generation**: Build convex hull phase diagrams to identify thermodynamically stable phases
- **SQLite Database Storage**: Persistent storage of all generated structures and metadata
- **Incremental Exploration**: Load and reuse structures from previous runs to accelerate exploration
- **Parallel Execution**: Run multiple chemical system explorations in parallel

## Installation

### From source
```bash
git clone https://github.com/ourofoundation/ggen.git
cd ggen
pip install -e .
```

### Development installation
```bash
git clone https://github.com/ourofoundation/ggen.git
cd ggen
pip install -e ".[dev,docs,jupyter]"
```

## Quick Start

### Basic Crystal Generation

```python
from ggen import GGen

# Initialize the generator
ggen = GGen()

# Generate a simple crystal structure
result = ggen.generate_crystal("SiO2", num_trials=5, optimize_geometry=True)

print(f"Generated: {result['name']}")
print(f"Space group: {result['final_space_group_symbol']} #{result['final_space_group']}")
print(f"Energy: {result['best_crystal_energy']:.4f} eV")

# Save as CIF
with open("sio2.cif", "w") as f:
    f.write(result['cif_content'])
```

### Structure Mutations

```python
# Load a structure
ggen.set_structure(structure)

# Apply various mutations
ggen.scale_lattice(1.1)  # Scale lattice by 10%
ggen.shear_lattice((2.0, -1.0, 0.5))  # Shear lattice angles
ggen.substitute("Si", "Ge", fraction=0.5)  # Substitute half the Si with Ge
ggen.jitter_sites(sigma=0.01)  # Add small random displacements

# Apply a sequence of mutations
mutations = [
    {"op": "scale_lattice", "scale_factor": 1.05},
    {"op": "substitute", "element_from": "O", "element_to": "S", "fraction": 0.1},
    {"op": "jitter_sites", "sigma": 0.005}
]
ggen.mutate_crystal(mutations, repair=True)
```

### Chemical Space Exploration

Systematically explore a chemical system and generate a phase diagram:

```python
from ggen import ChemistryExplorer

# Initialize the explorer
explorer = ChemistryExplorer(output_dir="./exploration_runs")

# Explore the Fe-Mn-Si system
result = explorer.explore(
    chemical_system="Fe-Mn-Si",
    max_atoms=16,          # Maximum atoms per unit cell
    num_trials=15,         # Generation attempts per stoichiometry
    optimize=True,         # Optimize geometries with MLIP
    preserve_symmetry=True # Maintain high symmetry during optimization
)

# Print results
print(f"Generated {result.num_successful} structures")
print(f"Found {len(result.hull_entries)} stable phases on the convex hull")

# Get stable/near-stable candidates
stable = explorer.get_stable_candidates(result, e_above_hull_cutoff=0.025)
for candidate in stable:
    print(f"{candidate.formula}: {candidate.energy_per_atom:.4f} eV/atom")

# Plot phase diagram (requires plotly)
fig = explorer.plot_phase_diagram(result)
fig.write_html("phase_diagram.html")

# Export summary
explorer.export_summary(result, "summary.json")
```

### Incremental Exploration with Previous Runs

Build on previous exploration results to accelerate new runs:

```python
explorer = ChemistryExplorer(output_dir="./exploration_runs")

# Load structures from previous runs and skip already-explored formulas
result = explorer.explore(
    chemical_system="Fe-Mn-Si",
    max_atoms=20,
    load_previous_runs=True,     # Load from all previous runs
    skip_existing_formulas=True  # Skip formulas already explored
)
```

### Trajectory Tracking

```python
# Enable trajectory tracking (default)
ggen = GGen(enable_trajectory=True)

# Generate and mutate structures
ggen.generate_crystal("BaTiO3")
ggen.scale_lattice(1.2)
ggen.symmetry_break()

# Export trajectory
ggen.export_trajectory_xyz("evolution.xyz")
ggen.export_trajectory_cif("evolution.cif")
ggen.export_trajectory_json("evolution.json")
ggen.export_trajectory_traj("evolution.traj")  # Standard ASE trajectory format

# Get trajectory data
trajectory = ggen.get_trajectory()
for frame in trajectory:
    print(f"Frame {frame['frame_index']}: {frame['composition']} - {frame['energy']} eV")
```

## Command Line Interface

### Explore a Single Chemical System

```bash
python scripts/explore_system.py Fe-Mn-Si
python scripts/explore_system.py Fe-Mn-Si --max-atoms 20 --num-trials 20
python scripts/explore_system.py Li-Co-O --crystal-systems tetragonal hexagonal
```

### Explore Multiple Systems in Parallel

```bash
python scripts/explore_parallel.py Fe-Mn-Si Li-Co-O Fe-Sn-B --workers 3
```

Or use GNU parallel:
```bash
parallel python scripts/explore_system.py ::: Fe-Mn-Si Li-Co-O Fe-Sn-B
```

## API Reference

### Core Classes

#### `GGen`
Main class for crystal generation and mutation operations.

**Initialization:**
```python
GGen(
    calculator=None,        # ASE calculator (default: ORB)
    random_seed=None,       # For reproducibility
    enable_trajectory=True  # Track structure evolution
)
```

**Key Methods:**
- `generate_crystal(formula, space_group=None, num_trials=10, optimize_geometry=False)`: Generate crystal structures
- `set_structure(structure, add_to_trajectory=True)`: Set the current structure
- `get_structure()`: Get the current structure
- `optimize_geometry(max_steps=400, fmax=0.01, relax_cell=True)`: Optimize structure geometry

**Mutation Methods:**
- `scale_lattice(scale_factor, isotropic=True)`: Scale lattice parameters
- `shear_lattice(angle_deltas)`: Modify lattice angles
- `substitute(element_from, element_to, fraction=1.0)`: Element substitution
- `add_site(element, coordinates, coords_are_cartesian=False)`: Add atomic sites
- `remove_site(site_indices=None, element=None, max_remove=None)`: Remove atomic sites
- `move_site(site_index, displacement=None, new_coordinates=None)`: Move atomic sites
- `jitter_sites(sigma=0.01, element=None)`: Add random displacements
- `symmetry_break(displacement_scale=0.01, angle_perturbation=0.1)`: Break symmetry
- `change_space_group(target_space_group, symprec=0.1)`: Change space group

**Analysis Methods:**
- `summary()`: Get comprehensive structure summary
- `describe_crystal_from_file(file_url, filename)`: Analyze structure from file
- `calculate_similarity(other_structure, method="fingerprint")`: Compare structures

**Trajectory Export Methods:**
- `export_trajectory_xyz(filename=None)`: Export trajectory as XYZ format
- `export_trajectory_cif(filename=None)`: Export trajectory as multi-block CIF
- `export_trajectory_json(filename=None)`: Export trajectory as JSON with metadata
- `export_trajectory_traj(filename=None)`: Export trajectory as standard ASE `.traj` format

---

#### `ChemistryExplorer`
Class for systematic exploration of chemical spaces and phase diagram generation.

**Initialization:**
```python
ChemistryExplorer(
    calculator=None,     # ASE calculator (default: ORB, lazily loaded)
    random_seed=None,    # For reproducibility
    output_dir=None      # Base directory for results (default: cwd)
)
```

**Main Methods:**
- `explore(chemical_system, ...)`: Run a full exploration of a chemical system
- `get_stable_candidates(result, e_above_hull_cutoff=0.025)`: Get stable/near-stable phases
- `plot_phase_diagram(result, show_unstable=0.1)`: Plot the convex hull phase diagram
- `export_summary(result, output_path=None)`: Export exploration summary as JSON

**Utility Methods:**
- `parse_chemical_system(system)`: Parse "Li-Co-O" → ["Co", "Li", "O"]
- `enumerate_stoichiometries(elements, max_atoms=12, ...)`: Generate candidate compositions
- `find_previous_runs(chemical_system)`: Find previous exploration directories
- `load_structures_from_previous_runs(chemical_system)`: Load best structures from previous runs
- `load_run(run_directory)`: Load a complete exploration result from disk

---

#### `CandidateResult`
Data class representing a generated structure candidate.

**Fields:**
- `formula`: Chemical formula (e.g., "Fe2MnSi")
- `stoichiometry`: Element counts dict (e.g., `{"Fe": 2, "Mn": 1, "Si": 1}`)
- `energy_per_atom`: Energy in eV/atom
- `total_energy`: Total energy in eV
- `num_atoms`: Number of atoms in the structure
- `space_group_number`: International space group number
- `space_group_symbol`: Space group symbol (e.g., "Fm-3m")
- `structure`: pymatgen Structure object
- `cif_path`: Path to saved CIF file
- `generation_metadata`: Additional metadata dict
- `is_valid`: Whether generation was successful
- `error_message`: Error message if generation failed

---

#### `ExplorationResult`
Data class containing complete results of a chemical space exploration.

**Fields:**
- `chemical_system`: System explored (e.g., "Fe-Mn-Si")
- `elements`: List of elements
- `num_candidates`: Total candidates attempted
- `num_successful`: Successful generations
- `num_failed`: Failed generations
- `candidates`: List of CandidateResult objects
- `phase_diagram`: pymatgen PhaseDiagram object
- `hull_entries`: List of candidates on the convex hull
- `run_directory`: Path to output directory
- `database_path`: Path to SQLite database
- `total_time_seconds`: Total exploration time

## Output Structure

Each exploration run creates a timestamped directory with:

```
exploration_Fe-Mn-Si_20260105_143500/
├── exploration.db      # SQLite database with all candidates
├── structures/         # CIF files for all generated structures
│   ├── Fe2Mn_Im-3m.cif
│   ├── FeMnSi_P6-mmm.cif
│   └── ...
├── phase_diagram.html  # Interactive phase diagram (plotly)
├── phase_diagram.png   # Static phase diagram image
└── summary.json        # JSON summary of results
```

## Dependencies

- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **pymatgen**: Materials analysis and structure manipulation
- **pyxtal**: Crystal structure generation
- **ase**: Atomic simulation environment
- **requests**: HTTP library for file downloads
- **orb-models**: ORB force field models for energy evaluation
- **pynanoflann**: Fast nearest neighbor search
- **plotly** (optional): Interactive phase diagram visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GGen in your research, please cite:

```bibtex
@software{ggen2025,
  title={GGen: Crystal Generation and Mutation Library},
  author={Matt Moderwell},
  year={2025},
  url={https://github.com/ourofoundation/ggen}
}
```

## Acknowledgments

- [PyXtal](https://github.com/qzhu2017/PyXtal) for crystal structure generation
- [pymatgen](https://github.com/materialsproject/pymatgen) for materials analysis
- [ASE](https://wiki.fysik.dtu.dk/ase/) for atomic simulation environment
- [ORB Models](https://github.com/orbital-materials/orb-models) for force field models

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.
