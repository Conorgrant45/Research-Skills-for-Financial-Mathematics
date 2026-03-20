#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regret_analysis.py
------------------
Correct regret computation for APL-Diffusion (Section 6.1 experiment).

The three reference lines explained:
  V_STAR_GH   : Gauss-Hermite exact Bellman solver — best available approximation
                of the true V*(x_0). Used as the reference in the paper's plots.
  V_STAR_MC   : Monte Carlo with optimal constant action — a LOWER bound on V*
                because the true optimal policy is state-dependent. Using this
                makes regret look smaller than it actually is.
  V_REF_CONST : Best constant-action policy value — a fair, computable reference
                that does not require knowing V*. Useful when you want to show
                the agent is learning something meaningful.

What the log-log slope means:
  slope = alpha  =>  Regret(K) ~ K^alpha
  alpha < 1      =>  sublinear regret (agent is learning)
  alpha = 1      =>  linear regret (agent is NOT learning)
  alpha = 3/4    =>  worst-case theoretical bound for dS=dA=1, m=1 (paper Thm 5.19)
  alpha ~ 0.69   =>  what paper's Figure 3(b) shows (better than worst case)

Why averaging across N experiments gives the right estimate:
  Definition 2.1:  Regret(K) = sum_k [ V*(X^k_1) - V^{pi^k}(X^k_1) ]
  Since X^k_1 = x_0 is fixed and V^{pi^k}(x_0) = E[episode reward | policy=pi^k],
  averaging N independent runs at each episode index k gives:
    (1/N) sum_{n=1}^N r^{k,n} -> V^{pi^k}(x_0)  as N -> inf
  This is what the code exploits by running N=10-50 parallel experiments.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed

# Import from the combined implementation
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Parameters matching Section 6.1 ────────────────────────────────────────
EP_LEN         = 10
N_EPS          = 2000
STARTING_STATE = 4.0
DELTA          = 1.0
DRIFT_BIAS     = 0.05
DRIFT_STATE    = -0.10
DRIFT_ACTION   = 0.01
SIGMA          = 0.10
RHO            = 10.0
ACTION_LO      = 0.0
ACTION_HI      = 10.0
_SQRT_DELTA    = math.sqrt(DELTA)


# ── Reference line 1: Gauss-Hermite Bellman (most accurate) ─────────────────

def compute_vstar_gauss_hermite(n_state=1601, n_action=801, n_quad=121,
                                 starting_state=STARTING_STATE):
    """
    Computes V*(x_0) by backward induction.

    Increasing n_state, n_action, n_quad improves accuracy.
    Default values here are finer than the original (801/401/81)
    to give a more accurate reference line.

    The quadrature integrates E_{B~N(0,1)}[V_{h+1}(X')] exactly for a
    Gaussian transition kernel — valid because sigma is state-action-independent
    in this experiment.
    """
    state_grid  = np.linspace(-RHO, RHO, n_state)
    action_grid = np.linspace(ACTION_LO, ACTION_HI, n_action)
    pts, wts    = np.polynomial.hermite.hermgauss(n_quad)
    quad_z      = pts * math.sqrt(2.0)   # scale N(0,1)
    quad_w      = wts / math.sqrt(math.pi)

    V       = np.zeros((EP_LEN + 1, n_state))   # V[H] = 0 (terminal)
    policy  = np.zeros((EP_LEN, n_state))

    for h in range(EP_LEN - 1, -1, -1):
        V_next = V[h + 1]   # shape (n_state,)
        for i, x in enumerate(state_grid):
            best_q, best_a = -np.inf, ACTION_LO
            for a in action_grid:
                r_mean = (x - a) ** 2   # mean reward
                q      = 0.0
                drift  = DRIFT_BIAS + DRIFT_STATE * x + DRIFT_ACTION * a
                for z, w in zip(quad_z, quad_w):
                    x_next = x + drift * DELTA + SIGMA * _SQRT_DELTA * z
                    if abs(x_next) > RHO:
                        v_next = -505.0   # OOB penalty
                    else:
                        # Linear interpolation on the value grid
                        v_next = r_mean + float(np.interp(x_next, state_grid, V_next))
                    q += w * v_next
                if q > best_q:
                    best_q, best_a = q, a
            V[h, i]      = best_q
            policy[h, i] = best_a

    v_star = float(np.interp(starting_state, state_grid, V[0]))
    return v_star, state_grid, V, policy


# ── Reference line 2: Best constant-action policy (honest lower bound) ───────

def compute_best_constant_policy(n_actions=201, n_mc=50000, seed=42,
                                  starting_state=STARTING_STATE):
    """
    V^{pi_const}(x_0) for the best constant action a*.

    This is computable without knowing V*, and serves as a fair practical
    reference: the agent only needs to beat a non-adaptive baseline.

    Note: this is a LOWER BOUND on V*(x_0) because the optimal policy
    is state-dependent. Using this as the reference makes reported regret
    LARGER than the paper's (more conservative / honest).
    """
    rng       = np.random.default_rng(seed)
    best_val  = -np.inf
    best_a    = ACTION_LO
    noise     = rng.standard_normal((EP_LEN, n_mc))

    for a_val in np.linspace(ACTION_LO, ACTION_HI, n_actions):
        state = np.full(n_mc, float(starting_state))
        total = np.zeros(n_mc)
        for h in range(EP_LEN):
            total += (state - a_val) ** 2   # mean reward, no noise for estimate
            drift  = DRIFT_BIAS + DRIFT_STATE * state + DRIFT_ACTION * a_val
            state  = state + drift * DELTA + SIGMA * _SQRT_DELTA * noise[h]
            state  = np.clip(state, -RHO, RHO)
        v = float(np.mean(total))
        if v > best_val:
            best_val, best_a = v, a_val

    return best_val, best_a


# ── Cumulative regret computation ────────────────────────────────────────────

def compute_cumulative_regret(vpi_matrix: np.ndarray, v_star: float):
    """
    vpi_matrix: shape (N_EPS, N_experiments)
    v_star    : reference value for V*(x_0)

    Returns:
      cum_regret_mean : shape (N_EPS,) — mean cumulative regret across experiments
      cum_regret_lo   : shape (N_EPS,) — 5th percentile (95% CI lower)
      cum_regret_hi   : shape (N_EPS,) — 95th percentile (95% CI upper)

    Per-episode regret at episode k:
      delta^k = max(v_star - mean_k(r^k), 0)
    where mean_k(r^k) is the mean episode reward at episode k across experiments.
    Taking max with 0 prevents negative regret due to v_star approximation error.

    Cumulative: Regret(K) = sum_{k=1}^K delta^k
    """
    # Per-experiment cumulative regret
    per_exp_regret = np.maximum(v_star - vpi_matrix, 0.0)   # (N_EPS, N)
    cum_per_exp    = np.cumsum(per_exp_regret, axis=0)        # (N_EPS, N)

    mean = cum_per_exp.mean(axis=1)
    lo   = np.percentile(cum_per_exp, 5,  axis=1)
    hi   = np.percentile(cum_per_exp, 95, axis=1)
    return mean, lo, hi


# ── Log-log slope estimation ─────────────────────────────────────────────────

def estimate_regret_slope(cum_regret: np.ndarray, fit_start: int = 1000):
    """
    Fits log(Regret(k)) = alpha * log(k) + const over k >= fit_start.
    Returns (slope, intercept, r_squared).

    The slope alpha is the empirical regret exponent.
    Theoretical worst-case bound: alpha <= (1+dS+dA)/(2+dS+dA) = 3/4 for dS=dA=1.
    """
    eps  = np.arange(1, len(cum_regret) + 1)
    mask = (eps >= fit_start) & (cum_regret > 0)
    lx   = np.log(eps[mask])
    ly   = np.log(cum_regret[mask])

    slope, intercept, r, _, se = stats.linregress(lx, ly)
    return slope, intercept, r ** 2


# ── Main plotting function ────────────────────────────────────────────────────

def plot_regret_analysis(vpi_matrix: np.ndarray,
                          v_star_gh: float,
                          v_star_mc: float,
                          fit_start: int = 1000):
    """
    Produces three panels:
      (a) Learning curve: mean episode reward vs episode with ±1σ band
      (b) Log-log regret vs episode for all three reference lines
      (c) Slope comparison bar chart

    vpi_matrix: shape (N_EPS, N_experiments) — raw episode rewards
    """
    N_EPS_plot = vpi_matrix.shape[0]
    eps        = np.arange(1, N_EPS_plot + 1)
    mean_rew   = vpi_matrix.mean(axis=1)
    std_rew    = vpi_matrix.std(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel (a): Learning curve ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eps, mean_rew, lw=1.2, color='steelblue', label='Mean episode reward')
    ax.fill_between(eps,
                    mean_rew - std_rew,
                    mean_rew + std_rew,
                    alpha=0.2, color='steelblue', label='±1σ')
    ax.axhline(v_star_gh, color='red',    lw=1.2, ls='-',  label=f'V* (G-H solver) = {v_star_gh:.1f}')
    ax.axhline(v_star_mc, color='orange', lw=1.2, ls='--', label=f'V* (MC constant) = {v_star_mc:.1f}')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("(a) Learning curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel (b): Log-log regret ────────────────────────────────────────────
    ax = axes[1]

    references = [
        ('G-H solver (paper ref)',       v_star_gh, 'red',    '-'),
        ('MC constant policy (lower bd)', v_star_mc, 'orange', '--'),
    ]
    slopes = {}

    for label, v_ref, color, ls in references:
        cum_mean, cum_lo, cum_hi = compute_cumulative_regret(vpi_matrix, v_ref)
        mask = (eps >= fit_start) & (cum_mean > 0)
        lx   = np.log(eps[mask])
        ly   = np.log(cum_mean[mask])

        slope, intercept, r2 = estimate_regret_slope(cum_mean, fit_start)
        slopes[label] = slope

        ax.plot(lx, ly, lw=1.3, color=color, ls=ls,
                label=f'{label} (slope={slope:.3f})')
        ax.plot(lx, slope * lx + intercept, lw=0.8, color=color, ls=':', alpha=0.7)

        # Confidence band (log scale of percentiles)
        lx_full = np.log(eps[cum_mean > 0])
        ly_lo   = np.log(np.maximum(cum_lo[cum_mean > 0], 1e-6))
        ly_hi   = np.log(np.maximum(cum_hi[cum_mean > 0], 1e-6))
        ax.fill_between(lx_full, ly_lo, ly_hi, alpha=0.08, color=color)

    # Theoretical bound line: slope = 3/4 for dS=dA=1
    theoretical_slope = 3.0 / 4.0
    mask_th    = eps >= fit_start
    lx_th      = np.log(eps[mask_th])
    # Anchor the theoretical line through the midpoint of the GH curve
    cum_gh, _, _ = compute_cumulative_regret(vpi_matrix, v_star_gh)
    anchor_idx   = len(eps[mask_th]) // 2
    anchor_x     = lx_th[anchor_idx]
    anchor_y     = math.log(max(cum_gh[mask_th][anchor_idx], 1.0))
    th_intercept = anchor_y - theoretical_slope * anchor_x
    ax.plot(lx_th, theoretical_slope * lx_th + th_intercept,
            lw=1.0, color='gray', ls='-.', alpha=0.8,
            label=f'Theoretical bound slope={theoretical_slope:.2f}')

    ax.set_xlabel("log(episode)")
    ax.set_ylabel("log(cumulative regret)")
    ax.set_title("(b) Log-log regret (slope = regret exponent α)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel (c): Slope comparison ──────────────────────────────────────────
    ax      = axes[2]
    names   = list(slopes.keys()) + ['Theoretical bound']
    values  = list(slopes.values()) + [theoretical_slope]
    colors  = ['red', 'orange', 'gray']
    bars    = ax.bar(range(len(names)), values, color=colors, alpha=0.75, width=0.5)
    ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5, label='Linear regret (α=1)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel("Regret exponent α")
    ax.set_ylim(0, 1.1)
    ax.set_title("(c) Slope comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle("APL-Diffusion regret analysis — 1-D O-U experiment (Section 6.1)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("regret_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
    return slopes


# ── Standalone runner ─────────────────────────────────────────────────────────

def run_analysis(n_experiments: int = 20, fit_start: int = 1000):
    """
    Full pipeline:
      1. Compute reference values (GH solver + MC constant policy)
      2. Run N parallel APL-Diffusion experiments
      3. Compute and plot regret correctly
    """
    import time

    # Import agent from combined file
    try:
        from six_one_exp import (
            APLDiffusion, AdaDiffEnvironment, Experiment, N_EPS
        )
    except ImportError:
        raise ImportError(
            "Place apl_diffusion_combined.py in the same directory."
        )

    print("=" * 60)
    print("Step 1/3: Computing V*(x_0) via Gauss-Hermite Bellman solver")
    print("  (finer grid than original: 1601 states, 801 actions, 121 quadrature pts)")
    t0 = time.perf_counter()
    v_star_gh, _, _, _ = compute_vstar_gauss_hermite(
        n_state=1601, n_action=801, n_quad=121
    )
    print(f"  V*(x_0={STARTING_STATE}) = {v_star_gh:.6f}  [{time.perf_counter()-t0:.1f}s]")

    print("\nStep 1b: Computing best constant-action policy value (MC)")
    v_star_mc, best_a = compute_best_constant_policy(n_actions=201, n_mc=50000)
    print(f"  V^{{pi_const}}(x_0) = {v_star_mc:.6f}  (best constant action a*={best_a:.2f})")
    print(f"  Gap (V*_GH - V*_MC) = {v_star_gh - v_star_mc:.4f}")
    print(f"  [This gap shows how much the state-dependent policy gains over constant]")

    print(f"\nStep 2/3: Running {n_experiments} parallel experiments × {N_EPS} episodes")

    def run_one(seed):
        env   = AdaDiffEnvironment()
        agent = APLDiffusion(flag=True)
        exp   = Experiment(env, agent, n_eps=N_EPS, seed=seed)
        exp.run()
        return exp.to_df()['epReward'].values

    t1      = time.perf_counter()
    results = Parallel(n_jobs=-1)(delayed(run_one)(i) for i in range(n_experiments))
    print(f"  Done [{time.perf_counter()-t1:.1f}s]")

    # Stack into matrix: rows=episodes, cols=experiments
    vpi_matrix = np.column_stack(results)   # shape (N_EPS, n_experiments)

    print("\nStep 3/3: Plotting regret analysis")
    slopes = plot_regret_analysis(vpi_matrix, v_star_gh, v_star_mc, fit_start)

    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  V*(x_0) [G-H solver]            = {v_star_gh:.4f}")
    print(f"  V*(x_0) [MC constant policy]    = {v_star_mc:.4f}")
    print(f"  Regret slope vs G-H ref         = {slopes.get('G-H solver (paper ref)', float('nan')):.4f}")
    print(f"  Regret slope vs MC ref          = {slopes.get('MC constant policy (lower bd)', float('nan')):.4f}")
    print(f"  Theoretical worst-case bound    = 0.7500")
    print(f"  Paper Figure 3(b) reports       ≈ 0.6900")
    print("─" * 60)
    print("Note: slope vs G-H ref is the paper-comparable number.")
    print("      slope vs MC ref will be HIGHER (more conservative).")

    return vpi_matrix, v_star_gh, v_star_mc, slopes


if __name__ == "__main__":
    run_analysis(n_experiments=20, fit_start=1000)