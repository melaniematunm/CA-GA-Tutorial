#!/usr/bin/env python3
"""
main.py — Run or resume the GA from Mitchell et al. 1994.

Usage examples
--------------
# Run fresh for 100 generations (paper defaults):
    python main.py

# Run fresh for 200 generations:
    python main.py --generations 200

# Resume from latest checkpoint and run more generations:
    python main.py --resume --generations 200

# Resume from a specific generation checkpoint:
    python main.py --resume --from-gen 50 --generations 200

# Turn off crossover (Section 11 experiment):
    python main.py --no-crossover

# Just generate plots from the latest checkpoint:
    python main.py --plots-only

# Evaluate GKL benchmark rule and compare:
    python main.py --eval-gkl

Options
-------
--generations N      Total generations to run (default: 100)
--resume             Resume from latest checkpoint
--from-gen N         Resume from this specific generation checkpoint
--no-crossover       Disable crossover (mutation only, Section 11)
--population N       Population size (default: 100)
--elite N            Elite size (default: 20)
--ics N              ICs per fitness eval (default: 100)
--lattice N          Lattice size (default: 149)
--seed N             Random seed (default: 42)
--checkpoint-dir D   Checkpoint directory (default: checkpoints)
--results-dir D      Results directory (default: results)
--plots-dir D        Plots directory (default: plots)
--plots-only         Skip GA, just regenerate plots from latest checkpoint
--eval-gkl           Evaluate GKL rule and print its fitness
--quiet              Suppress per-generation output
"""
import argparse
import sys
import numpy as np
from pathlib import Path

from ga import run_ga, load_checkpoint, DEFAULT_CONFIG, PAPER_CONFIG
from ca import make_gkl_rule, evaluate_rule, generate_ics
from analysis import generate_all_plots


def parse_args():
    p = argparse.ArgumentParser(
        description="GA for density-classifying CAs (Mitchell et al. 1994)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--generations",     type=int,  default=100)
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--from-gen",        type=int,  default=None)
    p.add_argument("--no-crossover",    action="store_true")
    p.add_argument("--population",      type=int,  default=100)
    p.add_argument("--elite",           type=int,  default=20)
    p.add_argument("--ics",             type=int,  default=100)
    p.add_argument("--lattice",         type=int,  default=149)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--checkpoint-dir",  type=str,  default="checkpoints")
    p.add_argument("--results-dir",     type=str,  default="results")
    p.add_argument("--plots-dir",       type=str,  default="plots")
    p.add_argument("--plots-only",      action="store_true")
    p.add_argument("--eval-gkl",        action="store_true")
    p.add_argument("--paper-fidelity",  action="store_true",
                    help="Use exact paper params: I=100, M_mean=320 (~8s/gen, ~13min/100gens)")
    p.add_argument("--quiet",           action="store_true")
    return p.parse_args()


def eval_gkl(N: int = 149, I: int = 10000, M: int = 1490, seed: int = 0):
    print("\n=== GKL Rule Evaluation ===")
    gkl = make_gkl_rule()
    lam = gkl.sum() / len(gkl)
    print(f"  Rule length : {len(gkl)} bits")
    print(f"  λ           : {lam:.4f}  (expected: 0.5)")
    rng = np.random.default_rng(seed)
    ics = generate_ics(I, N, rng)
    fitness = evaluate_rule(gkl, ics, M)
    print(f"  F_{I} (N={N}) : {fitness:.4f}  (paper reports ≈ 0.972)")
    return fitness


def main():
    args = parse_args()

    # --- GKL benchmark evaluation only ---
    if args.eval_gkl:
        eval_gkl(N=args.lattice)
        sys.exit(0)

    # --- Plots only ---
    if args.plots_only:
        print("Loading latest checkpoint for plotting...")
        state = load_checkpoint(Path(args.checkpoint_dir))
        best_idx  = state["fitnesses"].argmax()
        best_rule = state["population"][best_idx]
        generate_all_plots(state["history"], best_rule,
                            plots_dir=args.plots_dir, seed=args.seed)
        sys.exit(0)

    # --- Build config ---
    base_cfg = PAPER_CONFIG if args.paper_fidelity else DEFAULT_CONFIG
    cfg = base_cfg.copy()
    cfg["G"]    = args.generations
    cfg["P"]    = args.population
    cfg["E"]    = args.elite
    cfg["N"]    = args.lattice
    cfg["seed"] = args.seed
    if not args.paper_fidelity:
        cfg["I"] = args.ics

    # Handle --from-gen: load that checkpoint but override G
    resume = args.resume or (args.from_gen is not None)
    if args.from_gen is not None:
        ckpt_path = Path(args.checkpoint_dir) / "latest.txt"
        ckpt_path.write_text(str(args.from_gen))
        print(f"Resuming from generation {args.from_gen}")

    print("=" * 60)
    print("  Mitchell et al. 1994 — GA for Density-Classifying CAs")
    print("=" * 60)
    print(f"  Generations : {cfg['G']}")
    print(f"  Population  : {cfg['P']}  (elite={cfg['E']})")
    print(f"  Lattice N   : {cfg['N']}  ICs/eval={cfg['I']}")
    print(f"  Mode        : {'PAPER-FIDELITY (I=100, M=320)' if args.paper_fidelity else 'FAST (I=30, M=150) — use --paper-fidelity for exact replication'}")
    print(f"  Crossover   : {'ON' if not args.no_crossover else 'OFF'}")
    print(f"  Resume      : {resume}")
    print(f"  Seed        : {cfg['seed']}")
    print("=" * 60)

    # --- Run GA ---
    result = run_ga(
        cfg            = cfg,
        checkpoint_dir = args.checkpoint_dir,
        results_dir    = args.results_dir,
        resume         = resume,
        use_crossover  = not args.no_crossover,
        verbose        = not args.quiet,
    )

    # --- Final summary ---
    best_idx  = result["fitnesses"].argmax()
    best_rule = result["population"][best_idx]
    best_fit  = result["fitnesses"][best_idx]
    lam       = best_rule.sum() / len(best_rule)

    print("\n" + "=" * 60)
    print("  Run complete!")
    print(f"  Best fitness (F_100): {best_fit:.4f}")
    print(f"  Best rule λ         : {lam:.4f}")
    print(f"  GKL benchmark       : ≈ 0.972 (F_10000)")
    print("=" * 60)

    # --- Generate plots ---
    print("\nGenerating plots...")
    generate_all_plots(result["history"], best_rule,
                        plots_dir=args.plots_dir, seed=args.seed)

    # --- Save best rule ---
    np.save(f"{args.results_dir}/best_rule.npy", best_rule)
    print(f"\nBest rule saved to: {args.results_dir}/best_rule.npy")
    print("To load: import numpy as np; rule = np.load('results/best_rule.npy')")


if __name__ == "__main__":
    main()
