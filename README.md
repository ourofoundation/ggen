# GGen: Crystal Generation and Mutation Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful Python library for crystal structure generation, mutation, and evolutionary optimization. Built on top of PyXtal, PyMatGen, and ASE, GGen provides an intuitive interface for generating, modifying, and analyzing crystal structures with built-in energy evaluation using ORB models.

## Features

- **Crystal Generation**: Generate crystal structures from chemical formulas using PyXtal
- **Space Group Analysis**: Automatic space group compatibility checking and selection
- **Structure Mutations**: Comprehensive set of mutation operations for evolutionary optimization
- **Energy Evaluation**: Built-in energy calculation using ORB force field models
- **Trajectory Tracking**: Track structure evolution through mutations and optimizations
- **Multiple Export Formats**: Export structures as CIF, XYZ, or JSON
- **Structure Validation**: Built-in validation and repair mechanisms

## Installation

### From PyPI (when available)
```bash
pip install ggen
```

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

# Get trajectory data
trajectory = ggen.get_trajectory()
for frame in trajectory:
    print(f"Frame {frame['frame_index']}: {frame['composition']} - {frame['energy']} eV")
```

## API Reference

### Core Classes

#### `GGen`
Main class for crystal generation and mutation operations.

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

## Examples

### Evolutionary Crystal Optimization

```python
import numpy as np
from ggen import GGen

def fitness_function(ggen):
    """Calculate fitness based on energy and structure quality."""
    structure = ggen.get_structure()
    if structure is None:
        return float('inf')
    
    # Calculate energy
    try:
        atoms = ggen._atoms_from_structure(structure)
        atoms.calc = ggen.calculator
        energy = atoms.get_potential_energy()
    except:
        return float('inf')
    
    # Penalize structures with overlaps
    min_dist = min(structure.get_distance(i, j) 
                   for i in range(len(structure)) 
                   for j in range(i+1, len(structure)))
    if min_dist < 0.8:
        energy += 1000  # Large penalty for overlaps
    
    return energy

# Initialize population
population_size = 20
population = []
for _ in range(population_size):
    ggen = GGen()
    ggen.generate_crystal("Si8O16", num_trials=3)
    population.append(ggen)

# Evolutionary optimization
for generation in range(50):
    # Evaluate fitness
    fitness_scores = [fitness_function(gg) for gg in population]
    
    # Select best individuals
    sorted_indices = np.argsort(fitness_scores)
    best_individuals = [population[i] for i in sorted_indices[:population_size//2]]
    
    # Create new generation
    new_population = best_individuals.copy()
    for _ in range(population_size - len(best_individuals)):
        parent = np.random.choice(best_individuals)
        child = parent.copy()
        
        # Apply random mutations
        mutations = [
            {"op": "scale_lattice", "scale_factor": np.random.uniform(0.9, 1.1)},
            {"op": "jitter_sites", "sigma": np.random.uniform(0.001, 0.01)},
            {"op": "shear_lattice", "angle_deltas": np.random.normal(0, 1, 3)}
        ]
        child.mutate_crystal(mutations, repair=True)
        new_population.append(child)
    
    population = new_population
    
    # Print best fitness
    best_fitness = min(fitness_scores)
    print(f"Generation {generation}: Best fitness = {best_fitness:.4f}")

# Get best structure
best_ggen = min(population, key=fitness_function)
best_structure = best_ggen.get_structure()
print(f"Best structure: {best_structure.composition.reduced_formula}")
```

### Structure Comparison and Analysis

```python
from ggen import GGen

# Generate two similar structures
ggen1 = GGen()
ggen1.generate_crystal("TiO2", space_group=136)  # Tetragonal

ggen2 = GGen()
ggen2.generate_crystal("TiO2", space_group=136)
ggen2.scale_lattice(1.05)
ggen2.jitter_sites(sigma=0.01)

# Compare structures
similarity = ggen1.compare_branches(ggen2, method="fingerprint")
print(f"Similarity: {similarity['similarity']:.2f}%")

# Get detailed summaries
summary1 = ggen1.summary()
summary2 = ggen2.summary()

print(f"Structure 1: {summary1['formula']} - {summary1['space_group']['symbol']}")
print(f"Structure 2: {summary2['formula']} - {summary2['space_group']['symbol']}")
```

## Dependencies

- **numpy**: Numerical computations
- **scipy**: Scientific computing
- **pymatgen**: Materials analysis and structure manipulation
- **pyxtal**: Crystal structure generation
- **ase**: Atomic simulation environment
- **requests**: HTTP library for file downloads
- **orb-models**: ORB force field models for energy evaluation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development

### Setting up development environment

```bash
git clone https://github.com/yourusername/ggen.git
cd ggen
pip install -e ".[dev,docs,jupyter]"
pre-commit install
```

### Running tests

```bash
pytest
```

### Code formatting

```bash
black ggen/
isort ggen/
```

### Type checking

```bash
mypy ggen/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GGen in your research, please cite:

```bibtex
@software{ggen2024,
  title={GGen: Crystal Generation and Mutation Library},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ggen}
}
```

## Acknowledgments

- [PyXtal](https://github.com/qzhu2017/PyXtal) for crystal structure generation
- [PyMatGen](https://github.com/materialsproject/pymatgen) for materials analysis
- [ASE](https://wiki.fysik.dtu.dk/ase/) for atomic simulation environment
- [ORB Models](https://github.com/Open-Catalyst-Project/Open-Catalyst-Project) for force field models

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes and version history.
