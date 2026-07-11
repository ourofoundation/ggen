# GGen Credit Burn

A parallel structure-search campaign that drains a GPU credit budget on Modal by
running ggen exploration across many chemical systems at once, validating the
promising results with phonons, and reporting what's genuinely new versus the
Materials Project.

It exists to answer one practical question: *"I have ~$4k of Modal credits
expiring in N days — how do I turn that into durable scientific output instead of
just burned GPU-hours?"*

- **Campaign B — systematic ternary sweep.** Enumerate every binary and ternary
  over a fixed element palette, generate + relax candidate crystals (PyXtal +
  ORB via torch-sim), and build convex hulls.
- **Campaign C — phonon + novelty backfill.** Run phonon (dynamical-stability)
  calculations on near-hull candidates, then diff the combined hull against MP to
  surface new compositions and phases that beat MP's best.

Everything accumulates into the shared `ggen.db` on the Modal volume, which is
permanent and reusable long after the credits are gone.

---

## What it's made of

| Piece | Where | Role |
|---|---|---|
| `explore_one` | `scripts/burn.py` | GPU worker. Explores **one** system on a private `/tmp` DB seeded from lower-arity results; returns a small **shard** of only-new structures. Scales out to many containers. Never writes the master DB. |
| `phonons_batch` | `scripts/burn.py` | GPU worker. Validates a batch of near-hull structures; returns results. |
| `MasterDB` | `scripts/burn.py` | Singleton (`max_containers=1`) that owns the master `ggen.db`. The driver calls it serially to merge shards, snapshot seeds, track spend, apply phonons, and report discoveries — so master writes never race. |
| `main` (driver) | `scripts/burn.py` | Local entrypoint. Stages the sweep, streams results, enforces the budget. **Idempotent** — rerun to resume. |
| `merge_from` / `extract_subset` | `ggen/database.py` | Shard-level DB merge (dedup by `id` + `structure_hash`) and only-new extraction. |
| `summarize` | `ggen/discoveries.py` | New-vs-MP analysis, shared by the burn report and `scripts/discoveries.py`. |

### Why it's safe to fan out

The database is SQLite on a Modal volume, and volumes don't give reliable
cross-container file locking. So **only `MasterDB` ever writes the master DB**,
and it's pinned to a single container. Workers read a static seed snapshot and
write their own shard; the driver merges shards through `MasterDB` one at a time.

### Why it doesn't waste compute

Exploration is **staged** — unaries, then binaries (seeded with the unaries),
then ternaries (seeded with binaries + unaries). With `require_all_elements` each
phase only generates *its* arity; lower-arity endpoints come from the seed
instead of being recomputed for every system that shares them.

---

## Prerequisites

- Modal CLI configured with the profile that holds the credits:
  ```bash
  export MODAL_PROFILE=ouro-users   # or whichever profile has the credits
  ```
- No secrets required. ORB model weights download from Hugging Face on first run
  and cache to the volume (`/vol/.cache`), so subsequent containers start fast.
- The campaign reads/writes volume **`ggen-data`**, file **`data/ggen.db`** — the
  same database the deployed `materials/apps/ggen` app uses.

---

## How to start

Always start with a dry run, then a smoke test, then the real thing.

### 1. Dry run — see the plan and cost estimate, launch nothing

```bash
modal run ggen/scripts/burn.py --dry-run
```

Prints the palette, the unary/binary/ternary counts, the spend cap, and a rough
GPU-hour / dollar estimate.

### 2. Smoke test — a tiny palette at low concurrency

```bash
GGEN_BURN_MAX_CONTAINERS=4 modal run ggen/scripts/burn.py --palette Fe,Co,Si,Mn,Ni
```

Confirms the whole pipeline works end-to-end (explore → shard → merge → phonons →
discoveries) before you commit real credits.

### 3. Full burn

```bash
modal run ggen/scripts/burn.py
```

Defaults: 20-element palette (1140 ternaries), A10G GPUs, 15 concurrent
containers, $3,700 spend cap. Run it under `nohup`/`tmux` so it survives your
laptop sleeping — but it's safe to kill and resume (see below).

### Knobs

GPU type and concurrency are set via **environment variables** (read when the
module is imported), because Modal 1.2 can't override them per-call:

| Env var | Default | Notes |
|---|---|---|
| `GGEN_BURN_GPU` | `A10G` | `T4` (cheap/slow), `A10G` (best $/throughput for ORB), `L40S`, `A100`, `H100` (drain credits fastest). |
| `GGEN_BURN_MAX_CONTAINERS` | `15` | Caps parallel GPUs → caps burn rate. ~15×A10G ≈ $400/day. |

Everything else is a CLI flag on `modal run ... --flag value`:

| Flag | Default | Meaning |
|---|---|---|
| `--stage` | `all` | `all` \| `explore` \| `unary` \| `binary` \| `ternary` \| `phonon` |
| `--palette` | 20-element default | Comma-separated symbols, e.g. `Fe,Co,Si,Mn,Ni` |
| `--max-atoms` | `16` | Max atoms per cell |
| `--num-trials` | `15` | Generation attempts per stoichiometry |
| `--max-steps` | `300` | Relaxation steps |
| `--max-stoichiometries` | `0` (all) | Cap stoichiometries per system |
| `--max-spend` | `3700.0` | Stop launching new work past this estimated spend |
| `--phonon-cutoff` | `0.05` | Validate structures within this many eV of the hull |
| `--phonon-limit` | `0` (all) | Cap phonon jobs |
| `--phonon-batch-size` | `8` | Structures per phonon container |
| `--round-size` | `40` | Systems per budget-check round |

### Sizing the credits

Modal **list** rates ($/GPU-hr): T4 $0.59, A10G $1.10, L40S $1.95, A100 $2.10,
H100 $3.95. The 20-element default sweep is roughly 1140 ternaries × ~2 GPU-hr ≈
~2,300 GPU-hr ≈ **~$2.5k on A10G** for exploration, leaving headroom under the
$3.7k cap for phonons.

> Caveat: the spend numbers are a **driver-side estimate** from accumulated
> GPU-seconds, and assume list (preemptible) pricing. Non-preemptible US capacity
> can be ~3.75×. Treat the [Modal dashboard](https://modal.com/) as the source of
> truth for actual spend.

---

## How to monitor a running burn

- **Live driver output** — each round prints structures added, GPU-hours, and
  running estimated spend, e.g.
  `round done: +312 structures, 28.4 GPU-hr, total $812.50`.
- **Modal dashboard** — real container counts, logs, and actual billed spend.
- **Spend ledger** — persisted on the volume at `burn/ledger.json`
  (`gpu_seconds`, `est_cost`, per-phase breakdown). Survives restarts.

---

## How to come back to results

### Resume an interrupted burn

Just rerun the exact same command. The driver checks which systems are already in
the master DB and skips them, so it picks up where it left off:

```bash
modal run ggen/scripts/burn.py
```

### Read the discoveries report

Every completed run ends with a server-side report — no local DB needed:

```
DISCOVERIES vs Materials Project (within 50 meV)
  124 unique non-elemental phases on hull across 1140 systems
    new compositions: 88   beat MP: 36
    phonon: 41 stable / 12 unstable / 71 untested
    symmetry: 109 higher-symmetry / 15 P1

  Most credible discoveries:
    Fe3Si          Fm-3m     E=-7.6000  [NEW] (phonon OK)  Fe-Si
    ...
```

To regenerate that report at any time without re-exploring:

```bash
modal run ggen/scripts/burn.py --stage phonon   # validates + reports
```

(If there's nothing left to validate, it just prints the report.)

### Pull the database locally for deep analysis

```bash
# Download the master DB from the Modal volume
modal volume get ggen-data data/ggen.db ./ggen.db

# Full per-system novelty report (text + JSON)
python ggen/scripts/discoveries.py --near-hull 0.05 -o discoveries.json

# Per-system report + interactive phase diagram
python ggen/scripts/report.py Fe-Si

# Export the best candidates as CIFs for DFT / follow-up
python ggen/scripts/export.py Fe-Co-Si -n 25 --max-ehull 0.05
```

### Where everything lives (volume `ggen-data`)

| Path (in volume) | Contents |
|---|---|
| `data/ggen.db` | The master database — all structures, hulls, phonon results. |
| `burn/ledger.json` | Estimated spend ledger. |
| `burn/seed_binary.db`, `burn/seed_ternary.db` | Phase seed snapshots (regenerated each run; safe to delete between campaigns). |
| `.cache/` | Cached ORB model weights. |

---

## Running just one campaign

```bash
# Only the exploration sweep (B), no phonons
modal run ggen/scripts/burn.py --stage explore

# Only phonon validation + novelty (C) on whatever is already in the DB
modal run ggen/scripts/burn.py --stage phonon

# Bigger GPUs to drain credits faster
GGEN_BURN_GPU=A100 modal run ggen/scripts/burn.py
```
