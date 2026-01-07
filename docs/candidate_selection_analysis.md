# Candidate Selection Analysis: Initial Energy vs Final Energy

## Summary

**Finding**: Initial (unrelaxed) energy is a poor predictor of final relaxed energy. Selecting candidates based on initial energy often misses the best structures.

**Solution**: Relax ALL candidate structures and select by final energy. Use torch-sim for GPU-batched parallel relaxation to make this computationally feasible.

---

## The Problem

### Original Approach

The original `generate_crystal()` workflow:
1. Generate N random structures across multiple spacegroups
2. Compute single-point energy for each (no relaxation)
3. Score by: `initial_energy + symmetry_bonus + wyckoff_penalty`
4. Select the lowest-scoring candidate
5. Only then do full relaxation on the winner

### Why This Fails

We ran an empirical test (`scripts/test_selection_heuristics.py`) generating 30 candidates for Fe₂O₃ and fully relaxing ALL of them:

| Predictor | Correlation with Final Energy |
|-----------|-------------------------------|
| **Initial energy** | **r = +0.007** (essentially zero!) |
| Force RMS | r = -0.261 |
| Force max | r = -0.247 |
| Stress norm | r = -0.175 |
| Mini-relax Δ | r = -0.216 |

**Key insight**: Initial energy has essentially **no predictive power** for final relaxed energy.

### Ranking Analysis

For Fe₂O₃ with 30 candidates:
- The **true best** candidate (lowest final energy) was ranked **16th out of 30** by initial energy
- The structure with lowest initial energy ended up **18th** by final energy

Example from logs:
```
Selected best by FINAL energy: SG 81, energy=-33.005 eV (was rank 16 by initial energy)
Initial energy heuristic would have picked SG 147 (final energy=-32.105 eV, rank 4)
```

The structures that relaxed to the lowest energies often had **high** initial energies:
- SG 81: Init E = -18.1 eV → Final E = -33.0 eV (dropped 15 eV!)
- SG 150: Init E = -8.1 eV → Final E = -33.0 eV

Meanwhile, structures with low initial energy often converged to **shallow local minima**.

### Why Negative Correlations for Forces/Stress?

The negative correlations for force RMS, force max, and stress suggest that structures with **lower initial forces are actually worse**. Interpretation: these structures are already near a (shallow) local minimum with little room to improve during relaxation.

---

## The Solution

### Relax All Candidates

Instead of pre-selecting by initial energy:
1. Generate N candidates
2. **Fully relax ALL of them**
3. Select the one with lowest **final** energy

### torch-sim Integration

To make relaxing all candidates computationally feasible, we integrated [torch-sim](https://github.com/TorchSim/torch-sim):

- **Batched GPU relaxation**: Relax multiple structures simultaneously
- **Automatic memory management**: Handles GPU memory efficiently
- **ORB model support**: Native wrapper for ORB models
- **FIRE optimizer**: Fast convergence with cell relaxation

```python
import torch_sim as ts
from torch_sim.models.orb import OrbModel

# Wrap ORB model for torch-sim
ts_model = OrbModel(raw_orb_model, compute_stress=True, device="cuda")

# Relax all structures in parallel
final_state = ts.optimize(
    system=list_of_atoms,
    model=ts_model,
    optimizer=ts.Optimizer.fire,
    autobatcher=True,  # Automatic GPU memory management
    init_kwargs={"cell_filter": ts.CellFilter.frechet},
)
```

### CPU Fallback

On CPU, torch-sim's autobatcher doesn't work (memory estimation is GPU-specific), so we fall back to sequential relaxation. Still correct results, just slower.

---

## Implementation

### New Parameter: `relax_all_trials`

Added to `GGen.generate_crystal()`:
```python
result = ggen.generate_crystal(
    formula="Fe2O3",
    num_trials=10,
    optimize_geometry=True,
    relax_all_trials=True,  # NEW: relax all, pick best by final energy
)
```

### Defaults

| Method | Default | Rationale |
|--------|---------|-----------|
| `GGen.generate_crystal()` | `False` | Backward compatibility |
| `ChemistryExplorer.explore()` | `True` | Better results for exploration |
| `explore.py` CLI | `True` | Better phase diagrams |

### CLI Usage

```bash
# New default (relax all trials)
python scripts/explore.py Fe-Mn-Si

# Legacy behavior (faster but worse)
python scripts/explore.py Fe-Mn-Si --no-relax-all
```

---

## Performance Considerations

### Compute Cost

| Approach | Relative Cost | Quality |
|----------|---------------|---------|
| Legacy (10 trials, pick 1) | 1x | Often wrong |
| Relax all (10 trials, GPU batched) | ~2-3x | Correct |
| Relax all (10 trials, CPU sequential) | ~10x | Correct |

The increased cost is justified by significantly better structure quality.

### Recommendations

1. **Use GPU** when available for batched relaxation
2. **Reduce num_trials** if needed (5-10 is often sufficient when relaxing all)
3. **Early stopping** (future work): Stop relaxations that are clearly converging to high-energy minima

---

## Key Takeaways

1. **Don't trust initial energy** for structure selection
2. **Forces/stress are also poor predictors** (and even negatively correlated!)
3. **Relaxation is essential** to find the true energy landscape
4. **Batched GPU relaxation** makes relaxing all candidates practical
5. **torch-sim** provides efficient batched relaxation with ORB models

---

## References

- torch-sim: https://github.com/TorchSim/torch-sim
- Test script: `scripts/test_selection_heuristics.py`
- Implementation: `ggen/ggen.py` (`_batch_relax_candidates_torchsim`)

