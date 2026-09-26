# Handoff: GA for Density-Classifying Cellular Automata
### Mitchell, Crutchfield & Hraber (1994) — Replication in Progress

**For the next Claude instance picking this up.**

---

## What this project is

A Python replication of the genetic algorithm (GA) experiments from:

> Mitchell, M., Crutchfield, J. P., & Hraber, P. T. (1994).
> *Evolving Cellular Automata to Perform Computations: Mechanisms and Impediments.*
> Physica D, 75, 361–391.

The GA evolves 1D binary (k=2, r=3) cellular automata to perform **density classification**:
given a random initial configuration of 0s and 1s, the CA must relax to all-0s if the
initial density of 1s was below 0.5, or to all-1s if above 0.5.

---

## Current state

- **30 generations completed** (out of the paper's 100)
- **Best fitness so far**: 0.94 (F_30 ICs) — paper's GKL benchmark is ~0.972 (F_10000)
- **Latest checkpoint**: `checkpoints/gen_0030/`
- **Key observation already visible**: λ of best rule has drifted to ~0.41 (below 0.5),
  confirming the paper's "symmetry breaking" phenomenon (Section 8, Figure 11)
- **Epoch status**: Run has progressed through Epochs 1–3 and is into Epoch 4
  (block-expanding strategy, ~0.9 fitness plateau)

---

## How to resume immediately

```bash
# Install deps (Python 3.9+ required)
pip install -r requirements.txt

# Verify everything works
python test_ca.py

# Resume from gen 30 and run to gen 100
python main.py --resume --generations 100

# Resume and run to gen 200 (more exploration)
python main.py --resume --generations 200

# Run with exact paper parameters (I=100, M=320) — slower but fully faithful
python main.py --resume --paper-fidelity --generations 100
```

**`--resume` always picks up from `checkpoints/latest.txt`** (currently gen 30).
No other flags needed — config is saved in the checkpoint.

---

## What to do next (suggested tasks)

### 1. Continue the run to gen 100 (minimum)
The paper runs 100 generations. We're at 30. Resume and run to completion.

```bash
python main.py --resume --generations 100
```

### 2. Run the no-crossover experiment (Section 11, Table 2)
The paper compared GA with and without crossover. Run a parallel experiment:

```bash
python main.py --no-crossover --generations 100 \
    --checkpoint-dir checkpoints_nocross \
    --results-dir results_nocross \
    --plots-dir plots_nocross
```

Expected: ~26% of runs reach Epoch 3 (vs. 92% with crossover). T2 (onset of
Epoch 2) should be ~66 gens vs. ~4 gens with crossover.

### 3. Evaluate GKL benchmark rule
```bash
python main.py --eval-gkl
```
Expected: F_10000 ≈ 0.972 on N=149. The evolved rules typically reach 0.883–0.936.

### 4. Run 50 independent trials (full replication of Table 2)
```bash
for i in $(seq 1 50); do
    python main.py --seed $i --quiet \
        --checkpoint-dir checkpoints_run$i \
        --results-dir results_run$i \
        --plots-dir plots_run$i &
done
```
Then aggregate results across runs to replicate Table 2 statistics.

### 5. Regenerate plots at any time
```bash
python main.py --plots-only
```

---

## File structure

```
ca_ga/
  HANDOFF.md          ← this file
  README.md           ← user-facing docs and all CLI options
  ca.py               ← CA engine: vectorised step, GKL rule, IC generation
  ga.py               ← GA: elite selection, crossover, mutation, checkpointing
  analysis.py         ← all plotting functions
  main.py             ← CLI entry point
  test_ca.py          ← smoke tests (run first!)
  requirements.txt    ← numpy, matplotlib only

  checkpoints/
    latest.txt        ← points to gen_0030 (current state)
    gen_0010/         ← checkpoint at generation 10
    gen_0020/         ← checkpoint at generation 20
    gen_0030/         ← checkpoint at generation 30 (LATEST)
      population.npy  ← (100, 128) uint8 — all 100 rule tables
      fitnesses.npy   ← (100,) float64 — F_30 fitness of each rule
      meta.json       ← config dict + full history list

  results/
    history.csv       ← per-generation stats: best_fitness, mean_fitness,
                         best_lambda, mean_lambda, std_lambda, wall_time
    best_rule.npy     ← (128,) uint8 — best rule found so far

  plots/
    fitness_history.png   ← Figure 3 equivalent: best & mean fitness vs gen
    lambda_trajectory.png ← λ of best rule and population mean over time
    spacetime_evolved.png ← Figure 6/7 equivalent: evolved rule space-time
    spacetime_gkl.png     ← Figure 1 equivalent: GKL benchmark space-time
    perf_evolved.png      ← Figure 9 equivalent: performance vs ρ₀
    perf_gkl.png          ← Figure 2 equivalent: GKL performance vs ρ₀
```

---

## Key concepts for interpreting results

**λ (lambda)**: Fraction of 1-output-bits in the rule table. The task requires λ ≈ 0.5
for a correct solution. The GA breaks this symmetry, producing low-ρ₀ specialists
(λ < 0.5) or high-ρ₀ specialists (λ > 0.5). This is the central "impediment" of
the paper (Section 12).

**Four epochs of innovation** (visible in `fitness_history.png`):
| Epoch | Fitness | Strategy |
|-------|---------|----------|
| 1 | ~0.50 | Always relax to all-0s or all-1s |
| 2 | 0.53–0.70 | + correctly classify extreme-density ICs |
| 3 | ~0.80 | Expand sufficiently large blocks |
| 4 | 0.90–0.95 | Refined block expansion; plateau |

**GKL benchmark** (F_10000 ≈ 0.972): Uses propagating signals (checkerboard/boundary
patterns) for global density estimation — a fundamentally different and superior
strategy. The GA rarely evolves GKL-like rules (only ~2 runs out of 50 in the paper).

---

## Loading the checkpoint manually

```python
import numpy as np, json
from pathlib import Path

# Load latest checkpoint
pop  = np.load("checkpoints/gen_0030/population.npy")   # (100, 128)
fits = np.load("checkpoints/gen_0030/fitnesses.npy")    # (100,)
with open("checkpoints/gen_0030/meta.json") as f:
    meta = json.load(f)

best_rule = pop[fits.argmax()]   # (128,) best rule so far
print(f"Best fitness: {fits.max():.4f}")
print(f"Best λ: {best_rule.mean():.4f}")

# Run the best rule on a test IC
from ca import run_ca
ic = np.random.randint(0, 2, 149, dtype='uint8')
final, steps, fixed = run_ca(best_rule, ic, max_steps=1490)
print(f"Final density: {final.mean():.3f}, converged: {fixed}")
```

---

## Performance notes

- **Fast mode** (default): I=30 ICs, M_mean=150 steps → ~1 sec/generation
- **Paper-fidelity mode** (`--paper-fidelity`): I=100, M_mean=320 → ~8 sec/generation
- All evaluation is vectorised with NumPy (no loops over ICs within a generation)
- Memory: ~(P × I × N) = 100 × 100 × 149 ≈ 1.5M uint8 per generation — negligible
