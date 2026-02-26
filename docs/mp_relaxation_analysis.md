# MP Relaxation Analysis: ORB vs DFT Structure Fidelity

**Date:** February 23, 2026
**Script:** `scripts/analyze_mp_relaxation_changes.py`
**Database:** `ggen.db` (186,287 MP structures with original metadata)

---

## Executive Summary

We imported 189,188 structures from Materials Project into the ggen database and relaxed all of them with ORB to enable direct energy comparison and proper phase diagram construction. Of the 186,287 that completed relaxation with original metadata preserved, **11.3% (20,961) changed space group number** — meaning ~89% of MP structures are structurally faithful under ORB relaxation. However, the failures are systematic: structures overwhelmingly collapse toward P1 (triclinic), and certain space groups and chemistries are disproportionately affected.

---

## Database Context

| Source     | Structures | With ORB Energy |
|------------|------------|-----------------|
| Alexandria | 1,305,000  | —               |
| ggen       | 205,247    | 205,247         |
| MP         | 189,188    | 186,287         |
| **Total**  | **1,699,435** | **391,534**  |

MP energy statistics (ORB, eV/atom):

| Metric | Value |
|--------|-------|
| Count  | 186,287 |
| Mean   | -5.97 |
| Min    | -14.29 |
| Max    | +36.30 |

Data spans January 6 – February 23, 2026 (~7 weeks of collection).

---

## Space Group Changes

### Overall

- **186,287** relaxed MP structures with original metadata
- **20,961 (11.3%)** changed space group number during ORB relaxation
- ~89% maintained their original symmetry

### By Number of Elements

| Category   | Changed | Total   | % Changed |
|------------|---------|---------|-----------|
| Unary      | 117     | 620     | 18.9%     |
| Binary     | 2,010   | 18,744  | 10.7%     |
| Ternary    | 8,010   | 77,895  | 10.3%     |
| Quaternary | 6,684   | 67,551  | 9.9%      |
| 5-ary      | 3,805   | 17,638  | 21.6%     |
| 6-ary      | 303     | 3,624   | 8.4%      |
| 7+ ary     | 32      | 215     | 14.9%     |

Unary and 5-ary compositions show elevated change rates. The unary result is notable — even elemental structures change symmetry ~19% of the time, likely reflecting polymorphic metals (U, etc.) where ORB's potential energy surface differs from DFT.

### Most Unstable Space Groups

Space groups with highest rate of change during relaxation:

| SG  | Symbol   | Changed | Total | % Changed |
|-----|----------|---------|-------|-----------|
| 16  | P222     | 456     | 613   | **74.4%** |
| 48  | Pnnn     | 719     | 1,360 | **52.9%** |
| 8   | Cm       | 875     | 4,257 | 20.6%     |
| 5   | C2       | 533     | 2,886 | 18.5%     |
| 123 | P4/mmm   | 799     | 4,367 | 18.3%     |
| 12  | C2/m     | 1,198   | 6,958 | 17.2%     |
| 11  | P2₁/m   | 387     | 2,399 | 16.1%     |
| 15  | C2/c     | 766     | 5,134 | 14.9%     |
| 2   | P-1      | 1,185   | 8,389 | 14.1%     |

Most stable space groups:

| SG  | Symbol   | Changed | Total  | % Changed |
|-----|----------|---------|--------|-----------|
| 225 | Fm-3m    | 334     | 10,558 | **3.2%**  |
| 166 | R-3m     | 396     | 5,064  | 7.8%      |
| 139 | I4/mmm   | 325     | 4,105  | 7.9%      |
| 1   | P1       | 758     | 9,373  | 8.1%      |

High-symmetry cubic structures (Fm-3m / rock salt / FCC) are almost always preserved. P1 is also stable by definition — it's already the lowest symmetry, so changes represent ORB finding *higher* symmetry than DFT.

### Dominant Transition: Collapse to P1

The most striking finding is where structures go when they change. The top 20 SG transitions by count:

| Original SG | Relaxed SG | Count |
|-------------|------------|-------|
| 2 (P-1)     | **1 (P1)** | 819   |
| 8 (Cm)      | **1 (P1)** | 601   |
| 16 (P222)   | **1 (P1)** | 396   |
| 12 (C2/m)   | **1 (P1)** | 373   |
| 14 (P2₁/c) | **1 (P1)** | 338   |
| 5 (C2)      | **1 (P1)** | 290   |
| 216 (F-43m) | **1 (P1)** | 264   |
| 15 (C2/c)   | 9 (Cc)     | 240   |
| 15 (C2/c)   | **1 (P1)** | 236   |
| 12 (C2/m)   | 2 (P-1)    | 221   |
| 14 (P2₁/c) | 7 (Pc)     | 207   |
| 38 (Amm2)   | 6 (Pm)     | 197   |
| 62 (Pnma)   | **1 (P1)** | 189   |
| 6 (Pm)      | **1 (P1)** | 184   |
| 12 (C2/m)   | 5 (C2)     | 182   |
| 12 (C2/m)   | 8 (Cm)     | 177   |
| 146 (R3)    | **1 (P1)** | 164   |
| 71 (Immm)   | **1 (P1)** | 162   |
| 38 (Amm2)   | **1 (P1)** | 159   |
| 1 (P1)      | 2 (P-1)    | 152   |

**Of the top 20 transitions, 13 are collapses to P1.** This is the dominant failure mode: ORB relaxation breaks symmetry that DFT preserves. This is consistent with our earlier [P1 symmetry investigation](p1_symmetry_investigation_report.md) — the ORB potential has a slightly different energy landscape that often lacks the symmetry-preserving saddle points of DFT.

The one notable exception: **P1 → P-1 (152 cases)** — ORB *finding* inversion symmetry that DFT didn't assign. These are likely cases where the DFT-relaxed structure had approximate inversion symmetry just below the detection threshold.

---

## Volume Changes

| Metric | Value |
|--------|-------|
| Structures with geometry | 186,287 |
| Mean volume change | +0.99% |
| Mean \|volume change\| | 4.98% |
| Stdev | 17.27% |

The slight positive bias (+0.99% mean) suggests ORB systematically predicts slightly larger equilibrium volumes than DFT — a known tendency of ML potentials trained predominantly on PBE data.

### Extreme Outliers

| Formula   | Volume Change |
|-----------|---------------|
| C₃N       | +542.8%       |
| SiBO₃     | +530.4%       |
| CeSe₂     | +428.7%       |
| CaBO₃     | +378.5%       |
| CsEr₆CI₁₂ | +376.7%      |
| BaBO₃     | +373.8%       |
| BSbO₃     | +372.6%       |
| RhN       | +364.2%       |
| RuN       | +361.6%       |
| PtC       | +360.9%       |

These are structures that "exploded" during relaxation — the ORB potential has no local minimum near the DFT geometry, so the structure expands dramatically. Several borates (SiBO₃, CaBO₃, BaBO₃, BSbO₃) appear, suggesting ORB may struggle with boron-oxygen frameworks. The nitrides (C₃N, RhN, RuN) and PtC are likely cases where the MP structure is a metastable high-pressure phase with no ORB-accessible basin at ambient conditions.

---

## Top Formulas by SG Change Rate

| Formula      | Changed | Total | % Changed |
|--------------|---------|-------|-----------|
| CeSe₂        | 40      | 89    | 44.9%     |
| VO₂          | 20      | 33    | 60.6%     |
| CrN₂         | 14      | 40    | 35.0%     |
| LaMnO₃       | 11      | 16    | 68.8%     |
| SiO₂         | 10      | 120   | 8.3%      |
| Fe₃O₄        | 9       | 20    | 45.0%     |
| MoO₂         | 9       | 15    | 60.0%     |
| TiO₂         | 9       | 27    | 33.3%     |
| AlV₂O₄       | 8       | 10    | 80.0%     |

These are overwhelmingly **transition metal compounds with competing polymorphs and/or Jahn-Teller active ions**: VO₂ (rutile ↔ monoclinic), LaMnO₃ (orthorhombic ↔ rhombohedral perovskite), TiO₂ (rutile ↔ anatase ↔ brookite), Fe₃O₄ (inverse spinel distortions). The ORB potential resolves these polymorphic competitions differently than DFT.

### Chemical Systems in MP

The MP import is dominated by lithium battery-relevant systems:

| Chemical System | Count |
|-----------------|-------|
| Li-Mn-O-P       | 240   |
| F-Fe-O           | 219   |
| Fe-Li-O-P        | 216   |
| Li-Mn-O          | 214   |
| F-Li-O-V         | 208   |
| Li-O-P-V         | 206   |

This reflects the Materials Project's coverage emphasis on energy storage materials.

---

## Implications for Phase Diagrams

### What's Safe

- **~89% of structures maintain symmetry** — for most MP entries, the ORB-relaxed structure is a reasonable representation of the same phase, just re-evaluated on the ORB potential.
- **High-symmetry cubic structures are very reliable** (Fm-3m: 96.8% retention). Phase diagrams for simple metal alloys and rock salt / perovskite systems should be trustworthy.
- **Mean volume change of ~5%** is within acceptable bounds for comparing energy rankings.

### What Needs Care

1. **Filter extreme volume changes.** Structures with |ΔV| > 50% should be excluded — they represent ORB relaxation failures, not meaningful phase comparisons. This likely affects a few hundred structures.

2. **SG-changed structures aren't necessarily wrong for phase diagrams.** If ORB relaxes SG 12 → SG 1, it may have found a *nearby* lower-energy configuration on the ORB surface. For hull construction, the energy is what matters — but the structure is now a different phase than what MP intended.

3. **Polymorphic systems need manual inspection.** Formulas like VO₂, TiO₂, LaMnO₃ with high change rates will have their polymorph ordering scrambled. Phase diagrams involving these should cross-reference against known experimental ground states.

4. **P222 and Pnnn structures are unreliable.** With 74% and 53% change rates respectively, these space groups should be flagged or down-weighted in any phase diagram analysis.

5. **Boron-containing and high-pressure phases** show extreme volume expansion and should be treated with extra caution.

---

## Recommendations

1. **For hull computation:** include all MP structures but add a `volume_change_filter` flag to exclude |ΔV| > 50% outliers.

2. **For reporting:** when showing MP-derived phases on a phase diagram, indicate which ones changed SG during relaxation (already trackable via `mp_original_space_group_number` in metadata).

3. **For validation:** compare ORB energy rankings against MP DFT energy rankings for key systems (the `mp_energy_per_atom` is preserved in metadata) to quantify ranking fidelity beyond just structural fidelity.

4. **For problematic chemistries:** consider using the MP DFT energy directly (stored as `mp_energy_per_atom`) instead of the ORB energy when the ORB relaxation clearly failed (extreme volume change or dramatic SG collapse from high symmetry).
