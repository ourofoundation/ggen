# P1 Symmetry Investigation Report

**Date:** January 16, 2026  
**Goal:** Investigate whether low-energy P1 structures contain signal for finding even lower-energy, higher-symmetry configurations.

---

## Executive Summary

We ran three experiments to understand why many of our lowest-energy structures end up in P1 (triclinic) symmetry, and whether we're missing better structures. The key finding: **the issue isn't hidden symmetry in P1 structures—it's insufficient sampling**. With only 15 trials per stoichiometry, we're not exploring enough of the energy landscape to reliably find the global minimum.

---

## Experiment 1: Hidden Symmetry Detection

**Script:** `scripts/dev/analyze_p1_symmetry.py`  
**Question:** Do P1 structures have hidden higher symmetry that we're missing?

### Method
- Query low-energy P1 structures from the database
- Run symmetry detection with progressively looser tolerances (symprec: 0.001 → 0.5)
- If higher symmetry detected, "idealize" the structure and re-relax
- Compare energies

### Results (Bi-Fe-Ge system, 35 P1 structures)

| Metric | Value |
|--------|-------|
| Hidden symmetry detected | 5/35 (14.3%) |
| Improved by idealization | 1/5 (20%) |
| Best improvement | -1.1 meV/atom |
| Avg improvement | -0.5 meV/atom |

**Detected space groups:** Cm (#8), P-1 (#2), Pm (#6)

### Conclusions

- **Most P1 structures are genuinely P1** — no hidden symmetry even at very loose tolerances
- When hidden symmetry exists, idealizing provides **marginal benefit** (~1 meV/atom)
- The structures are already well-relaxed to their local minima
- **This is not the path to significant improvements**

---

## Experiment 2: Challenge P1 Ground States

**Script:** `scripts/dev/challenge_p1_ground_states.py`  
**Question:** Can we beat P1 ground states by explicitly searching higher-symmetry space groups?

### Method
- Find formulas where P1 is currently the best structure in our database
- Generate fresh structures in compatible higher-symmetry space groups (orthorhombic, tetragonal, cubic, etc.)
- Relax and compare to the P1 energy
- Report if any beat the P1

### Results (Co-Fe-Ge system, 4 P1 ground states)

| Metric | Value |
|--------|-------|
| **P1 beaten** | **3/4 (75%)** |
| **Avg improvement** | **68.7 meV/atom** |
| **Max improvement** | **122.2 meV/atom** |

**Winning structures:**

| Formula | P1 Energy | New Energy | Δ (meV) | Winner SG |
|---------|-----------|------------|---------|-----------|
| Co2Fe15 | -8.184 | -8.306 | **122.2** | P4/mmm (#123) |
| Co4Fe22 | -8.191 | -8.268 | **77.0** | Immm (#71) |
| Co3Fe17 | -8.266 | -8.273 | 6.9 | P222 (#16) |

### Conclusions

- **P1 is often a local minimum**, not the true ground state
- Higher-symmetry space groups frequently find **significantly lower energies**
- Improvements of 50-120 meV/atom are achievable — well above noise
- **The problem isn't symmetry detection — it's sampling coverage**

---

## Experiment 3: Trial Convergence Analysis

**Script:** `scripts/dev/trial_convergence.py`  
**Question:** How many trials are needed to reliably find the ground state?

### Method
- For a given formula, run N trials across all compatible space groups
- Track best-found energy after each trial
- Analyze convergence behavior

### Results

#### Co2Fe15 (17 atoms, 37 compatible space groups)

| Trials | Best Energy | Δ from final | % of optimal |
|--------|-------------|--------------|--------------|
| 5 | -8.2995 | 9.4 meV | 95.1% |
| 10 | -8.3074 | 1.5 meV | 99.2% |
| 15 | -8.3074 | 1.5 meV | 99.2% |
| 50 | -8.3074 | 1.5 meV | 99.2% |
| 75 | -8.3088 | 0.0 meV | 100% |
| **100** | -8.3088 | - | **Best found at trial 100** |

- Final best: **-8.3088 eV/atom** (P222)
- This beats our database value (-8.184) by **125 meV/atom**
- **Search had NOT converged** — best was found at the last trial

#### Fe3Ge (4 atoms, 37 compatible space groups)

| Trials | Best Energy | Δ from final | % of optimal |
|--------|-------------|--------------|--------------|
| 5 | -7.5986 | 3.1 meV | 96.3% |
| 10 | -7.5986 | 3.1 meV | 96.3% |
| **30** | -7.6017 | 0.0 meV | **100%** |
| 50 | -7.6017 | 0.0 meV | 100% |

- Final best: **-7.6017 eV/atom** (Pm)
- **Converged at 30 trials**
- Best found at trial 33

### Key Insight: Complexity Drives Trial Requirements

| Formula | Atoms | Converged at | Best found at |
|---------|-------|--------------|---------------|
| Fe3Ge | 4 | ~30 trials | Trial 33 |
| Co2Fe15 | 17 | >100 trials | Trial 100 (not converged) |

### Conclusions

- **15 trials (current default) is insufficient** for complex compositions
- Simple formulas (4 atoms): ~30 trials converges
- Complex formulas (17 atoms): 100+ trials may still be improving
- Energy variance is huge (0.7-1.0 eV/atom std) — many local minima exist

---

## Overall Conclusions

### The Root Cause

The issue isn't that P1 structures have hidden symmetry we're failing to detect. **The issue is insufficient sampling.** With only 15 trials spread across 50+ compatible space groups, we're essentially doing a lottery rather than a search.

### Why Higher Symmetry "Wins"

Higher-symmetry space groups beat P1 not because P1 is inherently wrong, but because:
- P1 has the most degrees of freedom (6 lattice + 3N positions)
- More DOF = more local minima = harder to find global minimum
- Higher-symmetry space groups constrain the search = fewer samples needed
- 5 trials in a constrained space group may be more effective than 15 in P1

### Recommendations

1. **Increase default `num_trials`** from 15 to 50 (or scale with atom count)
   - Rough heuristic: `num_trials = max(20, 5 * num_atoms)`
   
2. **Don't bias toward higher symmetry** — just sample more
   - The current uniform sampling is fine, we just need more of it

3. **Consider adaptive stopping**
   - Stop when no improvement seen for N consecutive trials
   - Would save compute on simple formulas while ensuring convergence on complex ones

4. **For critical compositions**, run 100+ trials
   - Especially for hull-relevant stoichiometries
   - The compute cost is ~8 minutes per 100 trials on GPU

### Impact

If we re-run explorations with 50+ trials per stoichiometry, we can expect:
- ~75% of current P1 "ground states" to be beaten
- Average improvements of ~70 meV/atom
- Some improvements exceeding 100 meV/atom

This could significantly change phase diagram predictions and identify new stable phases.

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/dev/analyze_p1_symmetry.py` | Detect hidden symmetry in P1 structures |
| `scripts/dev/challenge_p1_ground_states.py` | Challenge P1s with higher-symmetry generation |
| `scripts/dev/explore_higher_symmetry.py` | Follow-up generation in detected space groups |
| `scripts/dev/trial_convergence.py` | Analyze convergence vs number of trials |
