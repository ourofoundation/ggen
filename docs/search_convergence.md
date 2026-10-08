# How far from converged is explore's random search?

**Date:** 2026-10-08
**Script:** `materials/apps/ggen/bench_search.py` (Modal, A10G), tables from `bench_search_analysis.py`
**Raw data:** `docs/data/search_convergence_2026-10-08/` (one JSON per formula, every relaxed structure as CIF)

## Question

`explore` gives every stoichiometry 15 random PyXtal trials spread over 5 space
groups, relaxes them all, and keeps the lowest energy. Is that enough to find
the lowest minimum the proposer can reach? If not, hull distances built from
those winners are unreliable, because an undersampled competitor makes a
mediocre phase look stable.

## Method

For each formula: 200 candidates proposed exactly as `explore` proposes them
(4 seeds × `generate_candidates(num_trials=50, top_k_spacegroups=5)`), all
relaxed with the pooled torch-sim relaxer (FIRE, 400 steps, fmax 0.01, ORB v3
conservative). For every trial we keep the final energy, the energy after 20,
50 and 100 steps from the same starting cell, the space group, and a basin id
from `StructureMatcher` (only pairs within 5 meV/atom are compared).

Five formulas are known compounds whose ORB-relaxed Materials Project ground
state is in the database, so they have a ground truth. Two are Fe-Co(-Mn)
alloy cells where the ground state is unknown.

Cost: about 2 GPU-minutes per formula; the whole study was under $1.

## Results

| Formula | Atoms | Best of 200 (eV/atom) | Reference (ORB) | Gap (meV) | Basins / 200 trials | Trials within 25 meV of best |
|---|---|---|---|---|---|---|
| Fe3Ge | 4 | -7.6018 Pm-3m | MP Pm-3m -7.6017 | 0 | 120 | 10% |
| Fe8B4 | 12 | -8.1784 I4/mcm | MP Fe2B -8.1782 | 0 | 143 | 4% |
| Fe4Sn8 | 12 | -5.4657 I4/mcm | MP FeSn2 -5.4654 | 0 | 158 | 4% |
| Fe12B4 | 16 | -8.2330 Pnma | MP Fe3B -8.2327 | 0 | 169 | 4% |
| Fe10Si6 | 16 | -7.7266 Cmcm | MP Fe5Si3 P6_3/mcm -7.687 | **-40** | 157 | 20% |
| Co2Fe15 | 17 | -8.3087 R-3m | ggen DB best P-1 -8.3065 | -2 | 179 | 8% |
| Co3Fe5Mn12 | 20 | -8.6630 P1 | ggen DB best P1 -8.6359 | **-27** | 197 | 10% |

### Best-of-n versus budget

Expected gap of best-of-n to best-of-200, in meV/atom, by resampling the 200
outcomes (90th percentile in parentheses), and the probability that best-of-n
is within 5 meV/atom of best-of-200:

| Formula | n=5 | n=10 | **n=15** | n=20 | n=30 | n=50 | n=75 | n=100 |
|---|---|---|---|---|---|---|---|---|
| Fe3Ge | 108 (421) | 24 (85) | **11 (32)** | 5 (24) | 2 (3) | 0 | 0 | 0 |
| Fe8B4 | 118 (200) | 78 (156) | **58 (133)** | 42 (119) | 25 (95) | 9 (44) | 2 | 0 |
| Fe4Sn8 | 78 (123) | 57 (104) | **42 (93)** | 31 (89) | 18 (71) | 6 | 1 | 0 |
| Fe12B4 | 89 (160) | 58 (124) | **40 (101)** | 30 (86) | 17 (70) | 6 (11) | 1 | 0 |
| Fe10Si6 | 45 (165) | 14 (15) | **7 (6)** | 5 (6) | 4 (6) | 3 (6) | 2 (5) | 1 (2) |
| Co2Fe15 | 82 (128) | 49 (122) | **30 (116)** | 19 (111) | 8 (9) | 2 | 1 | 0 |
| Co3Fe5Mn12 | 38 (66) | 23 (53) | **16 (40)** | 12 (29) | 8 (15) | 5 (10) | 3 (6) | 2 (6) |

| Formula | n=5 | n=10 | **n=15** | n=20 | n=30 | n=50 | n=75 | n=100 |
|---|---|---|---|---|---|---|---|---|
| Fe3Ge | 41% | 67% | **80%** | 89% | 97% | 100% | 100% | 100% |
| Fe8B4 | 18% | 31% | **43%** | 54% | 69% | 87% | 97% | 99% |
| Fe4Sn8 | 20% | 33% | **46%** | 58% | 73% | 91% | 98% | 100% |
| Fe12B4 | 16% | 28% | **40%** | 48% | 64% | 83% | 94% | 99% |
| Fe10Si6 | 15% | 27% | **39%** | 46% | 63% | 82% | 94% | 99% |
| Co2Fe15 | 24% | 44% | **60%** | 70% | 83% | 96% | 100% | 100% |
| Co3Fe5Mn12 | 5% | 9% | **14%** | 19% | 28% | 43% | 61% | 75% |

Emulating explore exactly (3 trials in each of the 5 space groups a seed
picks), the gap to best-of-200 for the four seeds was: Fe8B4 0, 0, 127, 133;
Fe4Sn8 0, 89, 0, 0; Fe12B4 0, 11, 113, 67. It is a lottery: either a seed
lands in the ground-state basin or it misses by more than 100 meV/atom.

### Where the ground state lives

For the known compounds the ground-state basin was hit by 3 to 5% of trials,
and almost nothing else is close: within 50 meV of the best there are 1 to 3
basins (Fe4Sn8: one basin, hit 8 times, nothing else within 50 meV). The
basin was reached from several *different* proposed space groups (Fe8B4 from
SG 79, 131, 165 and 84; Fe4Sn8 from 26, 38, 112, 162 and 74; Fe12B4 from 19
and 200). The proposed symmetry is not what decides the outcome; the basin
of attraction under relaxation is.

Spreading trials over more space groups helps a little and costs nothing.
Resampling 15 trials as 15 groups × 1 instead of 5 × 3 raises the chance of
landing within 5 meV from 29% to 42% for Fe12B4 and 43% to 47% for Fe8B4;
using every sampled group once (18 to 20 trials) gives 53 to 59%.

### Alloy cells are a different problem

Co3Fe5Mn12 found 197 basins in 200 trials, 37 of them within 50 meV of the
best, every one hit exactly once. Co2Fe15 had 13 low basins, 10 singletons.
These are site-ordering variants of a close-packed or bcc solid solution:
there is no single ground-state basin to converge to, the energy differences
between orderings are a few meV/atom, and random restart will keep finding
new ones indefinitely. The best-of-200 beat the database value by 27 meV/atom
for Co3Fe5Mn12, and the Co2Fe15 winner (-8.3087, R-3m by symprec 0.1) is the
same energy the January P222 result reached. Any claim about these
compositions should be made with a cluster expansion or special quasirandom
structures, not with "the lowest random cell we found".

### Fe10Si6

The best Fe10Si6 cell (Cmcm, -7.7266 eV/atom, 10.7 Å³/atom) is 40 meV/atom
below the ORB energy of the Materials Project Fe5Si3 phase. On the ORB Fe-Si
hull it sits 6 meV/atom above the Fe3Si (Fm-3m, -8.0131) to FeSi (P2_13,
-7.4522) tie line, so it is a near-hull ordered bcc-derived arrangement rather
than a new stable phase. MP lists Fe5Si3 at 34 meV above hull in DFT, so this
is consistent. Worth a look, not a discovery. Its CIF is in the raw data.

### Early pruning by energy does not work, and is not needed

Spearman correlation between the energy after k steps and the final energy:

| Formula | initial | step 20 | step 50 | step 100 |
|---|---|---|---|---|
| Fe3Ge | -0.02 | 0.54 | 0.80 | 0.97 |
| Fe8B4 | -0.27 | -0.09 | 0.10 | 0.62 |
| Fe4Sn8 | -0.02 | 0.08 | 0.16 | 0.56 |
| Fe12B4 | 0.00 | 0.01 | 0.13 | 0.36 |
| Fe10Si6 | 0.12 | 0.07 | 0.32 | 0.59 |
| Co2Fe15 | 0.00 | -0.10 | -0.10 | 0.15 |
| Co3Fe5Mn12 | -0.03 | -0.13 | -0.17 | -0.03 |

For cells of 12 atoms or more the ranking after 20 or 50 steps carries no
information, and the eventual winner is often near the bottom early (the
Fe8B4 winner ranked 192nd of 196 at step 20). The reason is visible in the
trajectories: trials that have converged by step 100 are the junk (mean gap
300 to 1000 meV/atom to the best), while the trials still descending are the
ones that end low. Since torch-sim already drops converged structures from
the batch, the compute goes to exactly the trials that matter. There is
nothing to prune. This closes the question left open in
`candidate_selection_analysis.md`.

### Other observations

- Inflated cells (volume more than 1.3× the winner's): 56% of trials at 4
  atoms, 10 to 16% at 12 to 16 atoms, 4% at 17 to 20. A starting-volume
  problem for small cells only.
- Unconverged after 400 steps: 0 to 2%.
- The raw new-basin rate never falls below 56%, because most basins are
  distinct junk. Restricted to trials within 50 meV of the best, the
  Good-Turing estimate of unseen basin mass is 0 to 0.22 for the compounds
  and 0.62 to 1.0 for the alloys. That restricted rate is a usable,
  prior-free "are we done here" signal.

## What this means for explore

1. **15 trials is roughly a coin flip at 12 to 16 atoms.** A known ground
   state is missed by more than 5 meV/atom about half the time, by more than
   100 meV/atom one time in ten. Hulls built at the default budget are not
   reliable at the cell sizes where novelty lives. About 75 trials gives 95%.
2. **Budget should scale with cell size, or adapt.** The restricted
   new-basin rate converges for compounds and never for alloys, which is the
   right behaviour: stop sampling a composition when its low-energy
   singletons run out, and treat alloy-like compositions differently.
3. **Spread trials over as many space groups as possible**, one trial each,
   instead of 5 groups × 3. Free improvement, no prior involved.
4. **Drop early pruning from the roadmap.**
5. **Memory-based search is untested here.** This study only measures
   random restart. The fact that the ground-state basin is reached from many
   different starting symmetries suggests it has a large basin of attraction,
   which is what minima hopping relies on; that is the next experiment.

## Reproduce

```bash
MODAL_PROFILE=ouro-users ~/.pyenv/versions/ouro/bin/modal run materials/apps/ggen/bench_search.py \
    --trials 200 --trials-per-seed 50 --partial-steps 20,50,100 --out /some/dir
~/.pyenv/versions/ouro/bin/python materials/apps/ggen/bench_search_analysis.py /some/dir --md tables.md
```
