# GA for Density-Classifying Cellular Automata
### Replication of Mitchell, Crutchfield & Hraber (1994)
*Physica D, 75, 361–391*

---

## What this code does

Evolves 1D binary (k=2, r=3) cellular automata using a genetic algorithm to perform
the **ρ_c = 1/2 density classification task**: given an initial configuration, the CA must
relax to all-0s if the initial density of 1s was below 0.5, or to all-1s if above 0.5.

The code faithfully replicates the paper's GA (Section 6) and all the phenomena described:
- Four epochs of innovation
- Symmetry breaking (λ drift away from 0.5)
- Combinatorial drift toward λ = 0.5
- GKL benchmark rule for comparison

---

## Setup

```bash
# Python 3.9+ required
pip install -r requirements.txt

# Verify everything works
python test_ca.py
```

---

## Quick start

### Run with paper defaults (100 generations)
```bash
python main.py
```

### Run for more generations
```bash
python main.py --generations 300
```

### Resume from where you left off (pick up seamlessly)
```bash
python main.py --resume
```
This loads the latest checkpoint from `checkpoints/` and continues running.
The total `--generations` count is the **new** target, not additional generations.
So to go from gen 100 to gen 200: `python main.py --resume --generations 200`

### Resume from a specific generation
```bash
python main.py --resume --from-gen 50 --generations 200
```

### Reproduce the no-crossover experiment (Section 11)
```bash
python main.py --no-crossover --generations 100
```

### Just regenerate plots from the latest checkpoint
```bash
python main.py --plots-only
```

### Evaluate the GKL benchmark rule
```bash
python main.py --eval-gkl
```

---

## All options

| Flag | Default | Description |
|------|---------|-------------|
| `--generations N` | 100 | Total generations to run |
| `--resume` | off | Resume from latest checkpoint |
| `--from-gen N` | latest | Resume from this specific checkpoint |
| `--no-crossover` | off | Mutation only (Section 11 experiment) |
| `--population N` | 100 | Population size P |
| `--elite N` | 20 | Elite size E |
| `--ics N` | 100 | ICs per fitness evaluation (I) |
| `--lattice N` | 149 | CA lattice size |
| `--seed N` | 42 | Random seed |
| `--checkpoint-dir D` | checkpoints/ | Where to save/load checkpoints |
| `--results-dir D` | results/ | Where to save CSV logs and best rule |
| `--plots-dir D` | plots/ | Where to save plots |
| `--plots-only` | off | Skip GA, regenerate plots only |
| `--eval-gkl` | off | Evaluate GKL benchmark rule |
| `--quiet` | off | Suppress per-generation output |

---

## Output files

```
checkpoints/
  latest.txt             ← pointer to most recent checkpoint generation
  gen_0010/
    population.npy       ← (P, 128) uint8 array of rule tables
    fitnesses.npy        ← (P,) float64 array of F_100 fitnesses
    meta.json            ← config, history, generation number

results/
  history.csv            ← per-generation stats (fitness, lambda, etc.)
  best_rule.npy          ← best rule found (128-bit numpy array)

plots/
  fitness_history.png    ← Figure 3 equivalent
  lambda_trajectory.png  ← lambda of best rule and population mean
  lambda_dist.png        ← Figure 11 equivalent (histogram mosaic)
  spacetime_evolved.png  ← Figure 6/7 equivalent for evolved best rule
  spacetime_gkl.png      ← Figure 1 equivalent for GKL rule
  perf_evolved.png       ← Figure 9 equivalent
  perf_gkl.png           ← Figure 2 equivalent
```

---

## Loading a saved rule in Python

```python
import numpy as np
from ca import run_ca, make_gkl_rule

# Load the best evolved rule
rule = np.load("results/best_rule.npy")

# Load GKL benchmark
gkl = make_gkl_rule()

# Run on a custom IC
ic = np.random.randint(0, 2, size=149, dtype='uint8')
final, steps, fixed_point = run_ca(rule, ic, max_steps=1490)
print(f"Final density: {final.mean():.3f}, steps: {steps}, fixed: {fixed_point}")
```

---

## Inspecting history

```python
import csv, json
import matplotlib.pyplot as plt

with open("results/history.csv") as f:
    rows = list(csv.DictReader(f))

gens = [int(r["generation"]) for r in rows]
best = [float(r["best_fitness"]) for r in rows]
plt.plot(gens, best)
plt.xlabel("Generation"); plt.ylabel("Best fitness")
plt.show()
```

---

## File structure

```
ca_ga/
  ca.py          ← CA simulation engine (apply_rule, run_ca, GKL rule)
  ga.py          ← Genetic algorithm (run_ga, checkpoint save/load)
  analysis.py    ← Plotting and analysis tools
  main.py        ← CLI entry point
  test_ca.py     ← Smoke tests
  requirements.txt
  README.md      ← This file
```

---

## Paper parameters (Section 6)

| Parameter | Value | Notes |
|-----------|-------|-------|
| k (states) | 2 | binary |
| r (radius) | 3 | 7-cell neighbourhood |
| N (lattice) | 149 | odd, to avoid ρ₀ = exactly 0.5 |
| P (population) | 100 | |
| E (elite) | 20 | top 20% copied unchanged |
| m (mutations) | 2 | bits flipped per offspring |
| G (generations) | 100 | |
| I (ICs/eval) | 100 | F_100 fitness function |
| M (max steps) | ~320 | Poisson-distributed, mean = 320 |
| Crossover | single-point | |
| Initial pop λ | uniform [0,1] | |

GKL benchmark: F_10000 ≈ 0.972. Typical GA best: F_10000 ≈ 0.883–0.936.

---

## Extending the experiment

To run the experiments from Table 2 (Section 11):

```bash
# Standard GA (92% reach Epoch 3)
python main.py --seed 1

# No crossover (26% reach Epoch 3)
python main.py --no-crossover --seed 1

# No crossover, initial pop peaked at λ=1/2 (44%)
# Edit DEFAULT_CONFIG in ga.py: set "initpop_half_lambda": True
# Then: python main.py --no-crossover --seed 1
```

To run 50 independent trials (as in the paper):
```bash
for i in $(seq 1 50); do
    python main.py --seed $i --checkpoint-dir checkpoints_$i \
                   --results-dir results_$i --plots-dir plots_$i --quiet
done
```
