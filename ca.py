"""
Cellular Automata engine — Mitchell et al. 1994
k=2, r=3, 1D, periodic boundary, density classification task.

Fully vectorised: evaluates all (rules × ICs) simultaneously.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Neighbourhood lookup (pre-compute indices once)
# ---------------------------------------------------------------------------
_NEIGH_CACHE = {}

def _get_neigh_indices(N: int, r: int = 3):
    key = (N, r)
    if key not in _NEIGH_CACHE:
        idx = np.arange(N)
        cols = [(idx + offset) % N for offset in range(-r, r + 1)]
        _NEIGH_CACHE[key] = np.stack(cols, axis=1)   # (N, 2r+1)
    return _NEIGH_CACHE[key]


def _neigh_to_index(configs: np.ndarray, neigh_idx: np.ndarray) -> np.ndarray:
    """
    configs  : (B, N) uint8
    neigh_idx: (N, 7) int
    returns  : (B, N) int -- rule-table indices
    """
    powers = (2 ** np.arange(6, -1, -1)).astype(np.int32)  # MSB first
    return (configs[:, neigh_idx] * powers).sum(axis=2)     # (B, N)


# ---------------------------------------------------------------------------
# Batch CA evaluation across all ICs
# ---------------------------------------------------------------------------

def batch_run_ca(rule: np.ndarray, ics_arr: np.ndarray, rhos: np.ndarray,
                  max_steps: int, r: int = 3) -> np.ndarray:
    """
    Run all ICs simultaneously under the given rule.

    Parameters
    ----------
    rule      : (128,) uint8
    ics_arr   : (I, N) uint8
    rhos      : (I,)   float
    max_steps : int

    Returns
    -------
    correct   : (I,) bool
    """
    I, N     = ics_arr.shape
    neigh    = _get_neigh_indices(N, r)
    configs  = ics_arr.copy()
    running  = np.ones(I, dtype=bool)
    finals   = np.full(I, -1, dtype=np.int8)

    for _ in range(max_steps):
        if not running.any():
            break
        run_idx = np.where(running)[0]
        lk      = _neigh_to_index(configs[run_idx], neigh)  # (|run|, N)
        nxt     = rule[lk]                                    # (|run|, N)

        is_zero = (nxt == 0).all(axis=1)
        is_one  = (nxt == 1).all(axis=1)
        settled = is_zero | is_one

        settled_global         = run_idx[settled]
        finals[settled_global] = is_one[settled].astype(np.int8)
        running[settled_global] = False
        configs[run_idx]        = nxt

    # ICs still running after max_steps: check final config
    still = np.where(running)[0]
    if still.size:
        finals[still] = (configs[still] == 1).all(axis=1).astype(np.int8)

    # Determine correctness
    low_mask  = rhos < 0.5
    high_mask = rhos > 0.5
    correct   = np.zeros(I, dtype=bool)
    correct[low_mask]  = finals[low_mask]  == 0
    correct[high_mask] = finals[high_mask] == 1
    return correct


# ---------------------------------------------------------------------------
# IC generation -- returns numpy arrays directly (faster than list of tuples)
# ---------------------------------------------------------------------------

def generate_ics(n_ics: int, N: int, rng: np.random.Generator):
    """Returns (ics_arr, rhos): (n_ics, N) uint8 and (n_ics,) float."""
    half = n_ics // 2
    rows, densities = [], []

    for _ in range(half):                          # low density
        rho = rng.uniform(0.0, 0.5)
        n1  = max(0, min(int(round(rho * N)), N // 2 - (1 if N % 2 else 0)))
        ic  = np.zeros(N, dtype=np.uint8); ic[:n1] = 1; rng.shuffle(ic)
        rows.append(ic); densities.append(n1 / N)

    for _ in range(n_ics - half):                  # high density
        rho = rng.uniform(0.5, 1.0)
        n1  = max(N // 2 + 1, min(int(round(rho * N)), N))
        ic  = np.zeros(N, dtype=np.uint8); ic[:n1] = 1; rng.shuffle(ic)
        rows.append(ic); densities.append(n1 / N)

    idx = rng.permutation(n_ics)
    return np.array(rows, dtype=np.uint8)[idx], np.array(densities)[idx]


# ---------------------------------------------------------------------------
# Single-rule helpers (for analysis / plotting)
# ---------------------------------------------------------------------------

def apply_rule(config: np.ndarray, rule: np.ndarray, r: int = 3) -> np.ndarray:
    N     = len(config)
    neigh = _get_neigh_indices(N, r)
    lk    = _neigh_to_index(config[np.newaxis], neigh)[0]
    return rule[lk]


def run_ca(rule: np.ndarray, ic: np.ndarray, max_steps: int, r: int = 3):
    config = ic.copy()
    N      = len(config)
    neigh  = _get_neigh_indices(N, r)
    all_z  = np.zeros(N, dtype=np.uint8)
    all_o  = np.ones(N,  dtype=np.uint8)
    for step in range(max_steps):
        if np.array_equal(config, all_z) or np.array_equal(config, all_o):
            return config, step, True
        lk     = _neigh_to_index(config[np.newaxis], neigh)[0]
        config = rule[lk]
    return config, max_steps, False


def get_lambda(rule: np.ndarray) -> float:
    return float(rule.sum()) / len(rule)


def evaluate_rule(rule: np.ndarray, ics, max_steps: int) -> float:
    """Accept (ics_arr, rhos) tuple or legacy list of (ic, rho) tuples."""
    if isinstance(ics, tuple):
        ics_arr, rhos = ics
    else:
        ics_arr = np.array([ic for ic, _ in ics], dtype=np.uint8)
        rhos    = np.array([r  for _, r in ics],  dtype=np.float64)
    return float(batch_run_ca(rule, ics_arr, rhos, max_steps).mean())


# ---------------------------------------------------------------------------
# GKL benchmark rule
# ---------------------------------------------------------------------------

def make_gkl_rule() -> np.ndarray:
    """Construct the Gacs-Kurdyumov-Levin rule (Section 5)."""
    rule = np.zeros(128, dtype=np.uint8)
    for idx in range(128):
        bits  = [(idx >> (6 - b)) & 1 for b in range(7)]
        si    = bits[3]
        votes = (bits[3] + bits[2] + bits[0]) if si == 0 else (bits[3] + bits[4] + bits[6])
        rule[idx] = 1 if votes >= 2 else 0
    return rule
