"""
Plotting and analysis tools — Mitchell et al. 1994 replication.

Generates:
  - Best fitness vs generation (Figure 3 equivalent)
  - Lambda distribution histograms (Figure 11 equivalent)
  - Space-time diagrams (Figures 4–8 equivalent)
  - Performance vs rho0 (Figures 2, 9 equivalent)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from ca import run_ca, evaluate_rule, generate_ics, get_lambda, make_gkl_rule


# ---------------------------------------------------------------------------
# 1. Fitness history plot
# ---------------------------------------------------------------------------

def plot_fitness_history(history: list, save_path: str = "plots/fitness_history.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    gens         = [h["generation"] for h in history]
    best_fitness = [h["best_fitness"] for h in history]
    mean_fitness = [h["mean_fitness"] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best_fitness, label="Best fitness",  color="steelblue",  lw=2)
    ax.plot(gens, mean_fitness, label="Mean fitness",  color="darkorange", lw=1.5, ls="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (F_100)")
    ax.set_title("Best and Mean Fitness vs Generation\n(Mitchell et al. 1994 — Figure 3 equivalent)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# 2. Lambda distribution mosaic
# ---------------------------------------------------------------------------

def plot_lambda_distribution(history: list, population_snapshots: dict,
                              save_path: str = "plots/lambda_dist.png"):
    """
    population_snapshots: dict mapping generation -> population array
    Shows lambda histograms at selected generations (Figure 11 equivalent).
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    gens_to_show = sorted(population_snapshots.keys())[:16]
    n = len(gens_to_show)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 2.8))
    axes = axes.flatten()

    bins = np.linspace(0, 1, 16)
    for i, gen in enumerate(gens_to_show):
        pop = population_snapshots[gen]
        lambdas = [get_lambda(r) for r in pop]
        axes[i].hist(lambdas, bins=bins, color="steelblue", edgecolor="white")
        axes[i].set_title(f"Gen {gen}", fontsize=9)
        axes[i].set_xlim(0, 1)
        axes[i].set_xlabel("λ", fontsize=8)
        axes[i].set_ylabel("Count", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Elite λ Distribution Over Generations\n(Figure 11 equivalent)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# 3. Space-time diagrams
# ---------------------------------------------------------------------------

def plot_spacetime(rule: np.ndarray, ic: np.ndarray, max_steps: int,
                   title: str = "", save_path: str = "plots/spacetime.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    N = len(ic)
    history = [ic.copy()]
    config  = ic.copy()
    all_zeros = np.zeros(N, dtype=np.uint8)
    all_ones  = np.ones(N,  dtype=np.uint8)

    from ca import apply_rule
    for _ in range(max_steps - 1):
        config = apply_rule(config, rule)
        history.append(config.copy())
        if np.array_equal(config, all_zeros) or np.array_equal(config, all_ones):
            break

    grid = np.array(history)  # (time, N)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(grid, cmap="binary", interpolation="nearest", aspect="auto")
    ax.set_xlabel("Site")
    ax.set_ylabel("Time →")
    rho = ic.mean()
    ax.set_title(f"{title}\nρ₀ ≈ {rho:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_spacetime_pair(rule: np.ndarray, N: int = 149, max_steps: int = 149,
                        title: str = "CA Space-Time",
                        save_path: str = "plots/spacetime_pair.png",
                        seed: int = 0):
    """Plot two space-time diagrams: one low-ρ0, one high-ρ0 IC."""
    rng = np.random.default_rng(seed)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    from ca import apply_rule

    def _run(rho_target):
        n1 = int(round(rho_target * N))
        ic = np.zeros(N, dtype=np.uint8)
        ic[:n1] = 1
        rng.shuffle(ic)
        hist = [ic.copy()]
        cfg = ic.copy()
        all0 = np.zeros(N, dtype=np.uint8)
        all1 = np.ones(N,  dtype=np.uint8)
        for _ in range(max_steps - 1):
            cfg = apply_rule(cfg, rule)
            hist.append(cfg.copy())
            if np.array_equal(cfg, all0) or np.array_equal(cfg, all1):
                break
        return np.array(hist), n1 / N

    grid_lo, rho_lo = _run(0.42)
    grid_hi, rho_hi = _run(0.58)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, grid, rho in zip(axes, [grid_lo, grid_hi], [rho_lo, rho_hi]):
        ax.imshow(grid, cmap="binary", interpolation="nearest", aspect="auto")
        ax.set_xlabel("Site")
        ax.set_ylabel("Time →")
        ax.set_title(f"ρ₀ ≈ {rho:.3f}")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# 4. Performance vs rho0
# ---------------------------------------------------------------------------

def plot_performance_vs_rho(rule: np.ndarray, N_list: list = None,
                             n_per_bin: int = 100, n_bins: int = 19,
                             M: int = None, title: str = "Performance vs ρ₀",
                             save_path: str = "plots/performance_vs_rho.png",
                             seed: int = 0):
    """Replicate Figures 2 and 9 from the paper."""
    if N_list is None:
        N_list = [149]
    if M is None:
        M = 10 * max(N_list)
    rng = np.random.default_rng(seed)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    from ca import run_ca as _run_ca

    rho_bins = np.linspace(0.0, 1.0, n_bins)
    fig, ax  = plt.subplots(figsize=(8, 5))
    colors   = ["steelblue", "darkorange", "green"]

    for N, color in zip(N_list, colors):
        fractions = []
        for rho_target in rho_bins:
            correct = 0
            for _ in range(n_per_bin):
                n1 = int(round(rho_target * N))
                ic = np.zeros(N, dtype=np.uint8)
                ic[:n1] = 1
                rng.shuffle(ic)
                actual_rho = n1 / N
                final, _, _ = _run_ca(rule, ic, M)
                if actual_rho < 0.5 and np.all(final == 0):
                    correct += 1
                elif actual_rho > 0.5 and np.all(final == 1):
                    correct += 1
                elif actual_rho == 0.5:
                    pass  # undefined
            fractions.append(correct / n_per_bin)
        ax.plot(rho_bins, fractions, label=f"N={N}", color=color, lw=2)

    ax.axvline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("ρ₀ (initial density)")
    ax.set_ylabel("Fraction correct")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# 5. Lambda trajectory scatter
# ---------------------------------------------------------------------------

def plot_lambda_trajectory(history: list,
                            save_path: str = "plots/lambda_trajectory.png"):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    gens      = [h["generation"] for h in history]
    lam_mean  = [h["mean_lambda"]  for h in history]
    lam_best  = [h["best_lambda"]  for h in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, lam_mean, label="Mean λ (population)", color="darkorange", lw=1.5)
    ax.plot(gens, lam_best, label="λ of best rule",       color="steelblue",  lw=1.5, ls="--")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("λ")
    ax.set_title("λ Trajectory — Symmetry Breaking Observable")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# 6. Full analysis report
# ---------------------------------------------------------------------------

def generate_all_plots(history: list, best_rule: np.ndarray,
                        plots_dir: str = "plots", seed: int = 0):
    plot_fitness_history(history, f"{plots_dir}/fitness_history.png")
    plot_lambda_trajectory(history, f"{plots_dir}/lambda_trajectory.png")

    # Space-time diagrams for the evolved best rule
    plot_spacetime_pair(best_rule, N=149, max_steps=149,
                        title="Best Evolved Rule — Space-Time",
                        save_path=f"{plots_dir}/spacetime_evolved.png", seed=seed)

    # Space-time diagrams for GKL benchmark
    gkl = make_gkl_rule()
    plot_spacetime_pair(gkl, N=149, max_steps=149,
                        title="GKL Rule — Space-Time (Benchmark)",
                        save_path=f"{plots_dir}/spacetime_gkl.png", seed=seed)

    # Performance vs rho0
    plot_performance_vs_rho(best_rule, N_list=[149], n_per_bin=30,
                             title="Evolved Best Rule — Performance vs ρ₀",
                             save_path=f"{plots_dir}/perf_evolved.png", seed=seed, M=600)
    plot_performance_vs_rho(gkl, N_list=[149],  n_per_bin=30,
                             title="GKL Rule — Performance vs ρ₀ (Figure 2 equivalent)",
                             save_path=f"{plots_dir}/perf_gkl.png", seed=seed, M=600)

    print(f"\nAll plots saved to: {plots_dir}/")
