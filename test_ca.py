#!/usr/bin/env python3
"""
test_ca.py — Smoke tests for the CA and GA implementation.
Run with: python test_ca.py
"""
import numpy as np
import sys

def test_gkl_rule():
    from ca import make_gkl_rule, run_ca
    gkl = make_gkl_rule()
    assert len(gkl) == 128, "Rule should be 128 bits"
    lam = gkl.sum() / len(gkl)
    assert abs(lam - 0.5) < 1e-9, f"GKL λ should be 0.5, got {lam}"

    # Test on a clearly low-density IC
    N = 149
    rng = np.random.default_rng(0)
    ic_low = np.zeros(N, dtype=np.uint8)
    ic_low[:20] = 1  # rho = 20/149 ≈ 0.134
    rng.shuffle(ic_low)
    final, steps, fixed = run_ca(gkl, ic_low, max_steps=1490)
    assert np.all(final == 0), f"GKL should classify low-density IC as 0 (got {final.mean():.3f})"

    # Test on a clearly high-density IC
    ic_hi = np.ones(N, dtype=np.uint8)
    ic_hi[:20] = 0   # rho = 129/149 ≈ 0.866
    rng.shuffle(ic_hi)
    final, steps, fixed = run_ca(gkl, ic_hi, max_steps=1490)
    assert np.all(final == 1), f"GKL should classify high-density IC as 1 (got {final.mean():.3f})"

    print("✓ GKL rule tests pass")


def test_ic_generation():
    from ca import generate_ics
    rng = np.random.default_rng(42)
    ics_arr, rhos = generate_ics(100, 149, rng)
    assert len(ics_arr) == 100
    low  = (rhos < 0.5).sum()
    high = (rhos > 0.5).sum()
    assert low == 50, f"Expected 50 low-rho ICs, got {low}"
    assert high == 50, f"Expected 50 high-rho ICs, got {high}"
    print("✓ IC generation tests pass")


def test_ga_one_generation():
    from ga import random_population, evaluate_population, next_generation, DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG.copy()
    cfg["P"] = 10
    cfg["E"] = 4
    cfg["I"] = 10
    rng = np.random.default_rng(1)
    pop = random_population(cfg["P"], rng)
    assert pop.shape == (10, 128)
    fits = evaluate_population(pop, cfg, rng)
    assert fits.shape == (10,)
    assert all(0 <= f <= 1 for f in fits)
    new_pop = next_generation(pop, fits, cfg, rng, use_crossover=True)
    assert new_pop.shape == (10, 128)
    print("✓ GA one-generation test pass")


def test_checkpoint_roundtrip():
    import tempfile, json
    from pathlib import Path
    from ga import save_checkpoint, load_checkpoint
    pop  = np.zeros((5, 128), dtype=np.uint8)
    fits = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    with tempfile.TemporaryDirectory() as tmpdir:
        state = {
            "generation": 5,
            "cfg": {"N": 149},
            "history": [{"generation": 5, "best_fitness": 0.5}],
            "population": pop,
            "fitnesses": fits,
        }
        save_checkpoint(state, Path(tmpdir), 5)
        loaded = load_checkpoint(Path(tmpdir))
        assert np.array_equal(loaded["population"], pop)
        assert np.array_equal(loaded["fitnesses"],  fits)
        assert loaded["generation"] == 5
    print("✓ Checkpoint round-trip test pass")


if __name__ == "__main__":
    errors = []
    for test_fn in [test_gkl_rule, test_ic_generation,
                    test_ga_one_generation, test_checkpoint_roundtrip]:
        try:
            test_fn()
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            errors.append(test_fn.__name__)

    if errors:
        print(f"\n{len(errors)} test(s) failed: {errors}")
        sys.exit(1)
    else:
        print("\nAll tests passed ✓")
