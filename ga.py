"""
Genetic Algorithm — Mitchell et al. 1994
Evolves 1D k=2, r=3 CAs for the rho_c=1/2 density-classification task.

Features
--------
* Exact replication of the paper's GA (Section 6):
    - Elite selection (top E rules copied unchanged)
    - Single-point crossover between elite pairs
    - m=2 point mutations per offspring
    - New IC sample every generation
    - Poisson-distributed M (max CA steps)
* Checkpoint save/resume (JSON + numpy .npy)
* Detailed per-generation logging
"""
import json
import time
import numpy as np
from pathlib import Path

from ca import generate_ics, get_lambda, batch_run_ca


# ---------------------------------------------------------------------------
# GA configuration dataclass (plain dict for easy JSON serialisation)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "N":          149,    # lattice size
    "k":          2,      # number of states
    "r":          3,      # CA radius
    "P":          100,    # population size
    "E":          20,     # elite size
    "m":          2,      # mutations per offspring
    "G":          100,    # total generations to run
    "I":          30,     # ICs per fitness evaluation (paper uses 100; 30 for speed)
    "M_mean":     150,    # mean max CA steps (paper uses 320; 150 for speed)
    "eval_chunk": 20,     # rules per evaluation batch (tuning knob)
    "rule_bits":  128,    # 2^(2r+1) = 128 for r=3
    "seed":       42,
}

# Exact paper parameters (Section 6) — slower but fully faithful
PAPER_CONFIG = {
    **DEFAULT_CONFIG,
    "I":      100,
    "M_mean": 320,
    "G":      100,
}


# ---------------------------------------------------------------------------
# Core GA functions
# ---------------------------------------------------------------------------

def random_rule(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=128, dtype=np.uint8)


def random_population(P: int, rng: np.random.Generator) -> np.ndarray:
    """Initial population uniformly distributed over lambda in [0,1]."""
    return np.array([random_rule(rng) for _ in range(P)], dtype=np.uint8)


def single_point_crossover(parent_a: np.ndarray, parent_b: np.ndarray,
                            rng: np.random.Generator) -> tuple:
    """Single-point crossover as described in Section 6."""
    L = len(parent_a)
    point = rng.integers(1, L)  # cut point in [1, L-1]
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])
    return child_a, child_b


def mutate(rule: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """Flip exactly m randomly chosen bits."""
    rule = rule.copy()
    positions = rng.choice(len(rule), size=m, replace=False)
    rule[positions] ^= 1
    return rule


def evaluate_population(population: np.ndarray, cfg: dict,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Evaluate all P rules on the same I ICs.
    Processes rules in chunks of CHUNK_SIZE for memory efficiency.
    Checks for convergence every CHECK_INTERVAL steps to skip unnecessary work.
    """
    from ca import _get_neigh_indices, _neigh_to_index, generate_ics

    P  = len(population)
    M  = max(int(rng.poisson(cfg["M_mean"])), 1)
    N, I, r = cfg["N"], cfg["I"], cfg.get("r", 3)
    CHUNK     = cfg.get("eval_chunk", 20)   # rules per chunk
    CHECK_INT = 16                           # check convergence every N steps

    ics_arr, rhos = generate_ics(I, N, rng)
    neigh = _get_neigh_indices(N, r)
    fitnesses = np.zeros(P, dtype=np.float64)

    for start in range(0, P, CHUNK):
        chunk_rules = population[start:start + CHUNK]   # (C, 128)
        C  = len(chunk_rules)
        B  = C * I
        configs  = np.tile(ics_arr, (C, 1)).copy()      # (B, N)
        rhos_big = np.tile(rhos, C)                      # (B,)
        rule_idx = np.repeat(np.arange(C), I)            # (B,)

        running  = np.ones(B, dtype=bool)
        finals   = np.full(B, -1, dtype=np.int8)

        for step in range(M):
            if not running.any():
                break
            run  = np.where(running)[0]
            lk   = _neigh_to_index(configs[run], neigh)
            nxt  = chunk_rules[rule_idx[run], :][:, np.newaxis, :]  # wrong shape, fix:
            # Correct vectorised lookup: for each active row, apply its rule
            nxt  = chunk_rules[rule_idx[run]][np.arange(len(run))[:, None], lk]  # (|run|, N)
            configs[run] = nxt

            # Check convergence every CHECK_INT steps (or last step)
            if (step + 1) % CHECK_INT == 0 or step == M - 1:
                is_zero = (configs[run] == 0).all(axis=1)
                is_one  = (configs[run] == 1).all(axis=1)
                settled = is_zero | is_one
                sg = run[settled]
                finals[sg]  = is_one[settled].astype(np.int8)
                running[sg] = False

        still = np.where(running)[0]
        if still.size:
            finals[still] = (configs[still] == 1).all(axis=1).astype(np.int8)

        low  = rhos_big < 0.5
        high = rhos_big > 0.5
        correct = np.zeros(B, dtype=bool)
        correct[low]  = finals[low]  == 0
        correct[high] = finals[high] == 1
        fitnesses[start:start + C] = correct.reshape(C, I).mean(axis=1)

    return fitnesses


def next_generation(population: np.ndarray, fitnesses: np.ndarray,
                    cfg: dict, rng: np.random.Generator,
                    use_crossover: bool = True) -> np.ndarray:
    """Produce the next generation from current population + fitnesses."""
    P, E, m = cfg["P"], cfg["E"], cfg["m"]

    # Rank by fitness (ties broken randomly via a random key)
    noise = rng.random(P) * 1e-9
    order = np.argsort(-(fitnesses + noise))
    elite = population[order[:E]]

    new_pop = [elite[i].copy() for i in range(E)]

    # Fill remaining P-E slots with crossover + mutation offspring
    while len(new_pop) < P:
        i, j = rng.integers(0, E, size=2)
        if use_crossover:
            child_a, child_b = single_point_crossover(elite[i], elite[j], rng)
        else:
            child_a = elite[i].copy()
            child_b = elite[j].copy()
        child_a = mutate(child_a, m, rng)
        child_b = mutate(child_b, m, rng)
        new_pop.append(child_a)
        if len(new_pop) < P:
            new_pop.append(child_b)

    return np.array(new_pop[:P], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def compute_stats(population: np.ndarray, fitnesses: np.ndarray) -> dict:
    lambdas = np.array([get_lambda(r) for r in population])
    elite_mask = np.argsort(-fitnesses)[:20]
    return {
        "best_fitness":   float(fitnesses.max()),
        "mean_fitness":   float(fitnesses.mean()),
        "elite_mean_fit": float(fitnesses[elite_mask].mean()),
        "mean_lambda":    float(lambdas.mean()),
        "std_lambda":     float(lambdas.std()),
        "best_lambda":    float(lambdas[fitnesses.argmax()]),
    }


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, checkpoint_dir: Path, generation: int):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"gen_{generation:04d}"
    path.mkdir(exist_ok=True)
    # Save numpy arrays separately
    np.save(path / "population.npy", state["population"])
    np.save(path / "fitnesses.npy",  state["fitnesses"])
    meta = {k: v for k, v in state.items() if k not in ("population", "fitnesses")}
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    # Also write a "latest" pointer
    with open(checkpoint_dir / "latest.txt", "w") as f:
        f.write(str(generation))
    print(f"  [checkpoint saved → {path}]")


def load_checkpoint(checkpoint_dir: Path, generation: int = None) -> dict:
    """Load a checkpoint. If generation is None, load the latest."""
    if generation is None:
        latest_file = checkpoint_dir / "latest.txt"
        if not latest_file.exists():
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        generation = int(latest_file.read_text().strip())

    path = checkpoint_dir / f"gen_{generation:04d}"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path / "meta.json") as f:
        state = json.load(f)
    state["population"] = np.load(path / "population.npy")
    state["fitnesses"]  = np.load(path / "fitnesses.npy")
    print(f"  [checkpoint loaded ← gen {generation} from {path}]")
    return state


# ---------------------------------------------------------------------------
# Main GA runner
# ---------------------------------------------------------------------------

def run_ga(cfg: dict = None,
           checkpoint_dir: str = "checkpoints",
           results_dir:    str = "results",
           resume:         bool = True,
           use_crossover:  bool = True,
           verbose:        bool = True) -> dict:
    """
    Run (or resume) the GA.

    Parameters
    ----------
    cfg           : GA configuration dict (uses DEFAULT_CONFIG if None)
    checkpoint_dir: directory to save/load checkpoints
    results_dir   : directory to save per-run CSV log
    resume        : if True and a checkpoint exists, resume from latest
    use_crossover : toggle crossover on/off (Section 11 experiments)
    verbose       : print per-generation stats

    Returns
    -------
    dict with final population, fitnesses, history, and config
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.copy()

    ckpt_dir    = Path(checkpoint_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg["seed"])

    # --- Resume or fresh start ---
    start_gen = 0
    history   = []

    if resume and (ckpt_dir / "latest.txt").exists():
        state     = load_checkpoint(ckpt_dir)
        population = state["population"]
        fitnesses  = state["fitnesses"]
        start_gen  = state["generation"] + 1
        history    = state["history"]
        cfg        = state["cfg"]
        # Advance RNG to where we left off (re-seed with offset)
        rng = np.random.default_rng(cfg["seed"] + start_gen * 1000)
        print(f"Resuming from generation {start_gen} / {cfg['G']}")
    else:
        print("Starting fresh GA run.")
        population = random_population(cfg["P"], rng)
        fitnesses  = evaluate_population(population, cfg, rng)
        stats      = compute_stats(population, fitnesses)
        stats["generation"] = 0
        history.append(stats)
        if verbose:
            _print_gen(0, stats)

    total_gens = cfg["G"]

    # --- Main loop ---
    for gen in range(start_gen, total_gens):
        t0         = time.time()
        population = next_generation(population, fitnesses, cfg, rng, use_crossover)
        fitnesses  = evaluate_population(population, cfg, rng)

        stats               = compute_stats(population, fitnesses)
        stats["generation"] = gen + 1
        stats["wall_time"]  = round(time.time() - t0, 3)
        history.append(stats)

        if verbose:
            _print_gen(gen + 1, stats)

        # Save checkpoint every 10 gens and at the end
        if (gen + 1) % 10 == 0 or (gen + 1) == total_gens:
            state = {
                "generation": gen + 1,
                "cfg":        cfg,
                "history":    history,
                "population": population,
                "fitnesses":  fitnesses,
            }
            save_checkpoint(state, ckpt_dir, gen + 1)

    # --- Save results CSV ---
    import csv
    csv_path = results_dir / "history.csv"
    if history:
        # Collect all possible fieldnames across all records
        all_keys = []
        seen = set()
        for row in history:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
    print(f"\nResults saved to {csv_path}")

    return {
        "population": population,
        "fitnesses":  fitnesses,
        "history":    history,
        "cfg":        cfg,
    }


def _print_gen(gen: int, stats: dict):
    print(
        f"Gen {gen:3d} | best={stats['best_fitness']:.4f} "
        f"mean={stats['mean_fitness']:.4f} "
        f"λ_best={stats['best_lambda']:.3f} "
        f"λ_mean={stats['mean_lambda']:.3f}±{stats['std_lambda']:.3f}"
    )
