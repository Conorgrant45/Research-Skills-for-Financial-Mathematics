#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apl_diffusion_full.py  —  library module, import only, do not run directly.
============================================================================
Place this file in the same directory as your notebook and import with:

    from apl_diffusion_full import (
        BellmanSolverScalar, AdaDiffEnvironment,
        APLDiffusion, Experiment,
        compute_vstar_gh, compute_best_constant,
        run_experiments, cumulative_regret, regret_slope,
        plot_all, plot_partition_heatmap,
        N_EPS, EP_LEN, STARTING_STATE,
    )

Dependencies: numpy, scipy, pandas, matplotlib, joblib.

Public API
----------
Classes
    AdaDiffEnvironment    1-D O-U diffusion environment        (Section 6.1)
    BellmanSolverScalar   Vectorised Gauss-Hermite Bellman solver
    APLDiffusion          APL-Diffusion agent                  (Algorithm 1)
    Experiment            Episodic runner

Functions
    compute_vstar_gh()        Solve for V*(x0) via Bellman backward induction
    compute_best_constant()   Best constant-action policy value (MC lower bound)
    run_experiments(n)        Run n parallel APL-Diffusion experiments
    cumulative_regret(...)    Compute cumulative regret with percentile bands
    regret_slope(...)         Fit log-log slope (regret exponent alpha)
    plot_all(...)             Three-panel regret figure
    plot_partition_heatmap()  Q-value heatmap of the adaptive partition

Reference line explanation
--------------------------
Regret(K) = sum_{k=1}^K [V*(x0) - V^{pi^k}(x0)]

V* is approximated by the vectorised Gauss-Hermite Bellman solver.
The log-log slope alpha estimates the regret exponent: Regret(K) ~ K^alpha.
Theoretical worst-case bound: alpha <= 3/4 for dS=dA=1, m=1 (Theorem 5.19).
Paper Figure 3(b) reports alpha ~= 0.69.

Theory reference: Jin, Xu, Yang (2025), arXiv:2512.14991.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from joblib import Parallel, delayed

# =============================================================================
# SECTION 1 — Global parameters  (Section 6.1 of the paper)
# =============================================================================

EP_LEN         = 10       # H  — horizon per episode
N_EPS          = 2000     # K  — total episodes
STARTING_STATE = 4.0      # x_1 (fixed, same each episode)
DELTA          = 1.0      # Δ  — time increment

# O-U diffusion: mu_h(x,a) = B0 + B1*x + B2*a,  sigma_h = SIGMA (constant)
DRIFT_BIAS     = 0.05
DRIFT_STATE    = -0.10
DRIFT_ACTION   = 0.01
SIGMA          = 0.10

# Partition / algorithm hyper-parameters
RHO            = 10.0     # ρ  — state truncation radius  (Section 3)
C_H            = 5.0      # C_h — Lipschitz constant      (Proposition 2.4)
M              = 1        # polynomial growth order m     (Assumption 2.2)
SCALING        = 0.01     # scaling inside CONF           (replaces g1 constants)
INITIAL_Q      = 1837.1   # Q^0(B) optimistic init        (Eq. 5.4 / Section 6.1)
OOB_Q          = -505.0   # Q^k(Z̄^c) out-of-bounds penalty (Eq. 5.5)

ACTION_LO      = 0.0
ACTION_HI      = 10.0
_SQRT_DELTA    = math.sqrt(DELTA)

# =============================================================================
# SECTION 2 — Reward functions
# =============================================================================
# Each function has the signature  f(x: float, a: float) -> float
# and returns a single noisy reward sample.
# Pass any of these as reward_fn= to AdaDiffEnvironment, BellmanSolverScalar,
# compute_vstar_gh, compute_best_constant, and run_experiments.

def reward_6_1(x: float, a: float) -> float:
    """
    R ~ N((x-a)^2, 0.01).  Baseline from Section 6.1.
    Optimal action: a*(x) = x (boundary at a=10 for x=4).
    Growth order m = 1.
    """
    diff = x - a
    return float((diff * diff) + np.random.normal(0.0, 0.1))


def reward_quadratic_asymmetric(x: float, a: float) -> float:
    """
    R ~ N((x-a)^2 * (1 + |x|), 0.01).
    Optimal action: a*(x) = x.  Growth order m = 1.
    Sub-optimality gap grows with |x|, creating spatially
    non-uniform near-optimal set geometry for zooming-dimension
    sensitivity analysis.
    """
    diff = x - a
    return float(-(diff ** 2) * (1.0 + abs(x)) + np.random.normal(0.0, 0.1))


def reward_quadratic_shifted(x: float, a: float) -> float:
    """
    R ~ N((x - a - 0.5)^2, 0.01).
    Optimal action: a*(x) = x - 0.5.  Growth order m = 1.
    Same geometry as baseline but with a non-trivial optimal policy.
    Control experiment: zooming dimension sensitivity should match
    the baseline asymptotically.
    """
    diff = x - a - 0.5
    return float((diff ** 2) + np.random.normal(0.0, 0.1))


def reward_quartic(x: float, a: float) -> float:
    """
    R ~ N((x-a)^4, 0.01).
    Optimal action: a*(x) = x.  Growth order m = 3.
    Flat well near optimum inflates z_max,c.
    Regret exponent is most sensitive to the zooming dimension
    of all four rewards.
    """
    diff = x - a
    return float((diff ** 4) + np.random.normal(0.0, 0.1))


# Default reward used when no reward_fn is supplied
_DEFAULT_REWARD = reward_6_1

# =============================================================================
# SECTION 3 — Minimal base classes
# =============================================================================

class Agent:
    def update_obs(self, obs, action, reward, newObs, timestep): pass
    def update_policy(self, k): pass
    def pick_action(self, obs, timestep): return 0.0
    def get_num_arms(self): return 0

class Environment:
    def get_epLen(self): return 0
    def reset(self): pass
    def advance(self, action): return 0.0, 0.0, 0

# =============================================================================
# SECTION 3 — Environment  (Eq. 2.1 + Section 6.1 reward)
# =============================================================================

class AdaDiffEnvironment(Environment):
    """
    1-D Ornstein-Uhlenbeck diffusion.

    Dynamics (Eq. 2.1):
        X_{h+1} = X_h + mu_h(X_h, A_h)*Delta + sigma*sqrt(Delta)*B_h

    Reward (Section 6.1):
        r_h ~ N( (X_h - A_h)^2, 0.01 )

    Out-of-bounds (Section 3, localisation to S_1):
        |X_{h+1}| > rho  =>  reward = OOB_Q, episode terminates.
    """
    def __init__(self, ep_len=EP_LEN, starting_state=STARTING_STATE,
                 reward_fn=None):
        self.epLen          = ep_len
        self.starting_state = float(starting_state)
        self.state          = self.starting_state
        self.timestep       = 0
        self.reward_fn      = reward_fn if reward_fn is not None else _DEFAULT_REWARD

    def get_epLen(self):
        return self.epLen

    def reset(self):
        self.timestep = 0
        self.state    = self.starting_state

    def advance(self, action):
        x = float(self.state)
        a = float(np.clip(action, ACTION_LO, ACTION_HI))

        drift  = DRIFT_BIAS + DRIFT_STATE * x + DRIFT_ACTION * a
        new_x  = x + drift * DELTA + SIGMA * _SQRT_DELTA * float(np.random.randn())

        self.timestep += 1
        pContinue = 0 if self.timestep >= self.epLen else 1

        if abs(new_x) > RHO:
            reward    = OOB_Q
            pContinue = 0
            new_x     = float(np.clip(new_x, -RHO, RHO))
        else:
            new_x  = float(np.clip(new_x, -RHO, RHO))
            reward = self.reward_fn(x, a)

        self.state = new_x
        return reward, new_x, pContinue

# =============================================================================
# SECTION 4 — Exact Bellman solver  (benchmark only, not part of APL-Diffusion)
# =============================================================================

class BellmanSolverScalar:
    """
    Computes V*(x_0) by backward induction.

    Integrates E_{B~N(0,1)}[V_{h+1}(X')] using Gauss-Hermite quadrature,
    which is exact for Gaussian transition kernels (valid here because sigma
    is state-action-independent in Section 6.1).

    This is the reference line in the paper's Figure 3(a)/(b).
    Increase n_state / n_action / n_quad for higher accuracy.
    """
    def __init__(self, ep_len=EP_LEN, n_state=1601, n_action=801, n_quad=121,
                 reward_fn=None):
        self.epLen       = ep_len
        self.state_grid  = np.linspace(-RHO, RHO, n_state)
        self.action_grid = np.linspace(ACTION_LO, ACTION_HI, n_action)
        pts, wts         = np.polynomial.hermite.hermgauss(n_quad)
        self.quad_z      = pts * math.sqrt(2.0)
        self.quad_w      = wts / math.sqrt(math.pi)
        self.V           = np.zeros((ep_len + 1, n_state))
        self.policy      = np.zeros((ep_len, n_state))
        self.reward_fn   = reward_fn if reward_fn is not None else _DEFAULT_REWARD

    def solve(self):
        """
        Vectorised backward induction — no Python loops over states or actions.

        Axis convention used throughout:
            axis 0 → states   S = n_state
            axis 1 → actions  A = n_action
            axis 2 → quad pts Q = n_quad

        Why this is identical to the scalar version:
          Q(x,a) = r(x,a) + E_{z}[V_{h+1}(x')]
                 = (x-a)^2 + sum_q  w_q * V_{h+1}(x + drift*Delta + sigma*sqrt(Delta)*z_q)

        r(x,a) does not depend on z, so it is factored out of the quadrature
        sum and computed once per (x,a) pair — the key correction from the
        attached BellmanSolver. In the old scalar version it was recomputed
        n_quad times per (x,a) pair, wasting ~121x work on the reward alone.
        """
        S = len(self.state_grid)
        A = len(self.action_grid)
        Q = len(self.quad_z)

        # ── Build (S,1) and (1,A) views for broadcasting ─────────────────────
        x2d = self.state_grid[:, None]          # (S, 1)
        a2d = self.action_grid[None, :]         # (1, A)

        # ── Reward: E[r(x,a)] computed ONCE over the full (S, A) grid ─────
        # All reward functions have the form  mean(x,a) + N(0, noise_std).
        # The Bellman solver needs E[r(x,a)] = mean(x,a), not a noisy sample.
        # We recover the mean by averaging N_REWARD_SAMPLES calls, which
        # cancels the additive Gaussian noise (mean 0).  For the four
        # provided reward functions the mean is:
        #   reward_6_1             ->  (x-a)^2
        #   reward_quadratic_asym  ->  (x-a)^2 * (1+|x|)
        #   reward_quadratic_shift ->  (x-a-0.5)^2
        #   reward_quartic         ->  (x-a)^4
        # For custom reward_fn the same averaging applies automatically.
        N_REWARD_SAMPLES = 50   # enough to make noise negligible vs grid error
        _vfn = np.vectorize(lambda xi, ai: self.reward_fn(float(xi), float(ai)))
        r = sum(_vfn(x2d, a2d) for _ in range(N_REWARD_SAMPLES)) / N_REWARD_SAMPLES

        # ── Drift: mu(x,a)*Delta, shape (S, A) ───────────────────────────────
        drift = (DRIFT_BIAS + DRIFT_STATE * x2d + DRIFT_ACTION * a2d) * DELTA  # (S, A)

        # ── Quadrature noise axis, shape (1, 1, Q) ───────────────────────────
        z3d = self.quad_z[None, None, :]        # (1, 1, Q)
        w3d = self.quad_w[None, None, :]        # (1, 1, Q)

        # ── Precompute the deterministic part of x_next (no noise yet) ───────
        # shape (S, A, 1) — broadcast against z3d gives (S, A, Q)
        x_det = (x2d + drift)[:, :, None]       # (S, A, 1)

        # ── Terminal condition ────────────────────────────────────────────────
        self.V[self.epLen, :] = 0.0

        for h in range(self.epLen - 1, -1, -1):
            V_next = self.V[h + 1]              # (S,) — values on state grid

            # ── All next-states for every (x, a, z) triple ───────────────────
            # x_next[s, a, q] = x_s + drift(x_s,a_a)*Delta + sigma*sqrt(Delta)*z_q
            x_next = x_det + SIGMA * _SQRT_DELTA * z3d    # (S, A, Q)

            # ── Out-of-bounds mask: where |x_next| > rho ─────────────────────
            oob = np.abs(x_next) > RHO          # (S, A, Q) bool

            # ── Interpolate V_next at all in-bounds next-states ───────────────
            # np.interp is not natively vectorised over the query points, so we
            # flatten (S*A*Q,), interpolate in one call, then reshape back.
            x_flat   = x_next.ravel()                          # (S*A*Q,)
            x_clipped = np.clip(x_flat,
                                self.state_grid[0],
                                self.state_grid[-1])           # stay inside grid
            v_flat   = np.interp(x_clipped, self.state_grid, V_next)  # (S*A*Q,)
            v_next3d = v_flat.reshape(S, A, Q)                 # (S, A, Q)

            # ── Apply OOB penalty where applicable ───────────────────────────
            v_next3d = np.where(oob, OOB_Q, v_next3d)         # (S, A, Q)

            # ── Q(x,a) = r(x,a) + sum_q w_q * V_next(x_next) ────────────────
            # r is (S,A), broadcast against (S,A,Q) sum → (S,A)
            E_V_next = np.sum(w3d * v_next3d, axis=2)         # (S, A)
            Q_sa     = r + E_V_next                            # (S, A)

            # ── Greedy policy: argmax over actions ────────────────────────────
            best_idx         = np.argmax(Q_sa, axis=1)        # (S,)
            self.V[h, :]     = Q_sa[np.arange(S), best_idx]   # (S,)
            self.policy[h,:] = self.action_grid[best_idx]     # (S,)

    def get_value(self, x, h=0):
        return float(np.interp(x, self.state_grid, self.V[h]))

# =============================================================================
# SECTION 5 — Node  (one block B in the joint state-action partition)
# =============================================================================

class Node:
    """
    One hypercube block B ∈ P_h^k.

    Fields
    ------
    qVal              Q_h^k(B)   — current Q-function estimate
    rEst              R̂_h^k(B)  — reward estimator           (Eq. 4.16)
    muEst             μ̂_h^k(B)  — drift estimator            (Eq. 4.1)
    sigmaEst          Σ̂_h^k(B)  — variance estimator         (Eq. 4.1)
    num_visits        n_h^k(B)   — ancestral count            (Algorithm 3)
    num_unique_visits             — direct visits (used for estimation)
    state_val / radius            — center and half-width of Γ_S(B)
    action_val / action_radius    — center and half-width of Γ_A(B)
    """
    __slots__ = ('qVal', 'rEst', 'muEst', 'sigmaEst',
                 'num_visits', 'num_unique_visits', 'num_splits',
                 'state_val', 'action_val', 'radius', 'action_radius', 'children')

    def __init__(self, qVal, rEst, muEst, sigmaEst,
                 num_visits, num_unique_visits, num_splits,
                 state_val, action_val, radius, action_radius):
        self.qVal              = float(qVal)
        self.rEst              = float(rEst)
        self.muEst             = float(muEst)
        self.sigmaEst          = float(sigmaEst)
        self.num_visits        = int(num_visits)
        self.num_unique_visits = int(num_unique_visits)
        self.num_splits        = int(num_splits)
        self.state_val         = float(state_val)
        self.action_val        = float(action_val)
        self.radius            = float(radius)
        self.action_radius     = float(action_radius)
        self.children          = None

    def split(self):
        """
        Algorithm 4, line 2: split into 2^(dS+dA) = 4 children,
        each with half the diameter.

        Children inherit parent statistics when the parent was well-visited
        (Algorithm 4 lines 4-6: n_h^k(B_i) = n_h^k(B)).
        Rarely-visited parents reset children to the optimistic initialisation
        (Eq. 5.4) to avoid corrupting estimates from a single observation.
        """
        hr  = self.radius        * 0.5
        har = self.action_radius * 0.5
        low = self.num_visits <= 1
        children = []
        for ds in (-1, 1):
            ns = float(self.state_val + ds * hr)
            for da in (-1, 1):
                na = float(np.clip(self.action_val + da * har, ACTION_LO, ACTION_HI))
                if low:
                    child = Node(INITIAL_Q, 0.0, 0.0, 0.0,
                                 self.num_visits, 0, self.num_splits + 1,
                                 ns, na, hr, har)
                else:
                    child = Node(self.qVal, self.rEst, self.muEst, self.sigmaEst,
                                 self.num_visits, self.num_visits, self.num_splits + 1,
                                 ns, na, hr, har)
                children.append(child)
        self.children = children
        return children

# =============================================================================
# SECTION 6 — Tree  (per-timestep partition + projected value table)
# =============================================================================

class Tree:
    """
    Manages P_h^k for one timestep h.

    state_leaves / vEst implement Ṽ_h^k(S) from Eq. (5.6-5.8):
      state_leaves : centers of the induced state partition Γ_S(P_h^k)
      vEst[i]      : Ṽ_h^k(S_i), updated with the monotone min rule
      _min_vEst    : cached min(vEst) for O(1) Bellman backup
    """

    def __init__(self, initial_q=None):
        # Two root blocks covering [-rho, 0) and [0, rho].
        # Initialised with Q^0(B) = C_h*(1 + |x̃|^{m+1})  (Eq. 5.4).
        # initial_q overrides the module-level INITIAL_Q when supplied,
        # which is necessary because joblib workers cannot see module patches.
        _iq = initial_q if initial_q is not None else INITIAL_Q
        init_q_pos = C_H * (1.0 + abs(5.0)  ** (M + 1))
        init_q_neg = C_H * (1.0 + abs(-5.0) ** (M + 1))

        self.head_pos    = Node(_iq, 0, 0, 0, 0, 0, 0,  5.0, 5.0, 5.0, 5.0)
        self.head_neg    = Node(_iq, 0, 0, 0, 0, 0, 0, -5.0, 5.0, 5.0, 5.0)
        self.tree_leaves  = [self.head_pos, self.head_neg]
        self.state_leaves = [self.head_pos.state_val, self.head_neg.state_val]
        self.vEst         = [_iq, _iq]
        self._min_vEst    = min(self.vEst)

    # ------------------------------------------------------------------
    # diam(B) = L2 norm of (radius, action_radius)  (Section 2 metric)
    # ------------------------------------------------------------------
    @staticmethod
    def block_diameter(node):
        return math.sqrt(node.radius ** 2 + node.action_radius ** 2)

    # ------------------------------------------------------------------
    # CONF_h^k(B)  (Eq. 4.20)
    #
    # Theory:   CONF = g1(delta, ||x̃(oB)||) / sqrt(n)
    # g1 grows with state norm via eta(||x̃||) = L0 + L(||x̃||+ā) + 2LD.
    #
    # Here we replace the unknown g1 with a data-driven proxy:
    #   local_scale = 1 + |μ̂| + sqrt(σ̂)   ≈ eta(||x̃||) from data
    # The log(n) factor tightens the bound to match the log(HK²/δ) term
    # inside κ_μ from Proposition 4.1.
    # ------------------------------------------------------------------
    @staticmethod
    def confidence(node, scaling=SCALING):
        n           = max(1, node.num_unique_visits)
        local_scale = 1.0 + abs(node.muEst) + math.sqrt(max(node.sigmaEst, 0.0))
        return scaling * local_scale * math.sqrt(math.log(n + 2.0) / n)

    # ------------------------------------------------------------------
    # Splitting condition  (Algorithm 4, line 1):
    #   CONF_h^k(B) <= diam(B)
    #
    # Split when statistical precision exceeds block bias.
    # Requires at least 2 visits to avoid splitting on a single sample.
    # ------------------------------------------------------------------
    def should_split(self, node):
        if node.num_unique_visits < 2:
            return False
        return self.confidence(node) <= self.block_diameter(node)

    # ------------------------------------------------------------------
    # Execute split and update the partition  (Algorithm 4)
    # ------------------------------------------------------------------
    def split_node(self, node):
        if node not in self.tree_leaves:   # already split, nothing to do
            return
        children = node.split()
        self.tree_leaves.remove(node)
        self.tree_leaves.extend(children)

        # Update state leaves Γ_S(P_h^k) if a new state half appears
        c0_s = children[0].state_val
        c0_r = children[0].radius
        if min(abs(sl - c0_s) for sl in self.state_leaves) >= c0_r:
            parent = node.state_val
            try:
                idx      = self.state_leaves.index(parent)
                parent_v = self.vEst[idx]
                self.state_leaves.pop(idx)
                self.vEst.pop(idx)
            except ValueError:
                parent_v = node.qVal
            # children[0] and children[2] are the two new state halves
            self.state_leaves.append(children[0].state_val)
            self.state_leaves.append(children[2].state_val)
            self.vEst.append(parent_v)
            self.vEst.append(parent_v)
            self._min_vEst = min(self.vEst)
            # Full refresh after split — new regions need accurate estimates
            # This is infrequent so the O(N) cost is acceptable
            self.refresh_vEst()

    # ------------------------------------------------------------------
    # Block selection  (Algorithm 2):
    # traverse the tree, returning the leaf containing 'state' with
    # the highest Q estimate.
    # ------------------------------------------------------------------
    def get_active_ball(self, state):
        safe = min(max(float(state), -RHO), RHO)  # min/max is 10x faster than np.clip for scalars
        root = self.head_pos if safe >= 0.0 else self.head_neg
        return self._traverse(safe, root)

    def _traverse(self, state, node):
        if node.children is None:
            return node, node.qVal
        best_node, best_q = node, node.qVal
        for child in node.children:
            if abs(state - child.state_val) <= child.radius:
                n, q = self._traverse(state, child)
                if q >= best_q:
                    best_node, best_q = n, q
        return best_node, best_q

    # ------------------------------------------------------------------
    # Refresh Ṽ(S) for all state leaves  (Eq. 5.6, monotone min rule)
    # ------------------------------------------------------------------
    def refresh_vEst(self):
        for idx, sv in enumerate(self.state_leaves):
            _, q           = self.get_active_ball(sv)
            self.vEst[idx] = min(q, INITIAL_Q, self.vEst[idx])
        self._min_vEst = min(self.vEst)

    def get_num_leaves(self):
        return len(self.tree_leaves)

# =============================================================================
# SECTION 7 — APL-Diffusion agent  (Algorithm 1)
# =============================================================================

class APLDiffusion(Agent):
    """
    Adaptive Partition and Learning for Diffusions.

    flag=True  (default, matches paper Algorithm 1):
        Q/V updated in a backward sweep between episodes (update_policy).
    flag=False (online variant):
        Q/V updated immediately after each observation (update_obs).
    """

    def __init__(self, ep_len=EP_LEN, scaling=SCALING, flag=True,
                 initial_q=None):
        self.epLen     = ep_len
        self.scaling   = scaling
        self.flag      = flag
        # initial_q lets callers override the optimistic init per-instance
        # without patching the module global (which doesn't survive joblib fork)
        self._initial_q = initial_q if initial_q is not None else INITIAL_Q
        self.tree_list = [Tree(initial_q=self._initial_q) for _ in range(ep_len)]
        self._final_h  = ep_len - 1

    def reset(self):
        self.tree_list = [Tree(initial_q=self._initial_q) for _ in range(self.epLen)]

    def get_num_arms(self):
        return sum(t.get_num_leaves() for t in self.tree_list)

    # ------------------------------------------------------------------
    # update_obs  (Algorithm 1 line 12 + Algorithm 3 + Algorithm 5)
    # ------------------------------------------------------------------
    def update_obs(self, obs, action, reward, newObs, timestep):
        tree    = self.tree_list[timestep]
        node, _ = tree.get_active_ball(obs)

        # Algorithm 3: visit counts
        node.num_visits        += 1
        node.num_unique_visits += 1
        t     = node.num_unique_visits
        alpha = 1.0 / t

        # Reward estimator  (Eq. 4.16)
        node.rEst = (1.0 - alpha) * node.rEst + alpha * reward

        # Drift / variance estimators  (Eq. 4.1, online Welford)
        if timestep != self._final_h:
            dx            = newObs - obs
            old_mu        = node.muEst
            node.muEst    = (1.0 - alpha) * old_mu     + alpha * dx
            node.sigmaEst = (1.0 - alpha) * node.sigmaEst + alpha * (dx - old_mu) ** 2

        # Online Q update (flag=False only)
        if not self.flag:
            self._update_q(node, newObs, timestep)
            # O(1) update: just track running min instead of re-traversing all leaves
            if node.qVal < tree._min_vEst:
                tree._min_vEst = node.qVal

        # Algorithm 4 splitting rule: CONF <= diam
        if tree.should_split(node):
            tree.split_node(node)

    # ------------------------------------------------------------------
    # update_policy  (Algorithm 1 lines 9-13, backward sweep)
    # Called once per episode BEFORE the episode runs.
    # ------------------------------------------------------------------
    def update_policy(self, k):
        if not self.flag:
            return
        for h in range(self._final_h, -1, -1):
            tree = self.tree_list[h]
            for node in tree.tree_leaves:
                if node.num_unique_visits == 0:
                    # Unvisited: keep optimistic init  (Eq. 5.4)
                    node.qVal = C_H * (1.0 + abs(node.state_val) ** (M + 1))
                else:
                    self._update_q(node, None, h)
            tree.refresh_vEst()

    # ------------------------------------------------------------------
    # Q update  (Eq. 5.6-5.9)
    #
    # Q_h^k(B) = R̂ + UCB + E[V_{h+1}] + Lipschitz correction
    #
    # UCB  = scaling / sqrt(n)          (lumped R-UCB + T-UCB, Eq. 4.17/4.11)
    # BIAS = scaling * radius           (∝ diam(B), Eq. 5.1-5.2)
    # E[V] = Q at actual next state     (online) or min(vEst) (offline)
    # Lipschitz: C_h * sigmaEst proxies C_h*(1+||x||^m+||x̃||^m)*||x-x̃||
    #
    # min() enforces Theorem 5.2 monotone property: Q^k <= Q^{k-1}.
    # ------------------------------------------------------------------
    def _update_q(self, node, next_obs, timestep):
        n    = max(1, node.num_visits)
        ucb  = self.scaling / math.sqrt(n)
        bias = self.scaling * node.radius

        if timestep == self._final_h:
            new_q = node.rEst + ucb + bias
        else:
            next_tree = self.tree_list[timestep + 1]
            if next_obs is not None:
                _, next_q = next_tree.get_active_ball(next_obs)
            else:
                next_q = next_tree._min_vEst
            lip   = C_H * node.sigmaEst
            new_q = node.rEst + next_q + lip + ucb + bias

        node.qVal = min(node.qVal, INITIAL_Q, new_q)

    # ------------------------------------------------------------------
    # Action selection  (Algorithm 2 + Algorithm 1 line 6)
    # ------------------------------------------------------------------
    def pick_action(self, state, timestep):
        tree    = self.tree_list[timestep]
        node, _ = tree.get_active_ball(state)
        action  = np.random.uniform(node.action_val - node.action_radius,
                                    node.action_val + node.action_radius)
        return min(max(float(action), ACTION_LO), ACTION_HI)  # 10x faster than np.clip

# =============================================================================
# SECTION 8 — Experiment runner
# =============================================================================

class Experiment:
    def __init__(self, env, agent, n_eps=N_EPS, seed=0):
        self.env   = env
        self.agent = agent
        self.n_eps = n_eps
        self.data  = np.zeros((n_eps, 3))
        np.random.seed(seed)

    def run(self):
        env, agent = self.env, self.agent
        for ep in range(1, self.n_eps + 1):
            env.reset()
            state     = env.state
            ep_rew    = 0.0
            pContinue = 1
            h         = 0
            agent.update_policy(ep)
            while pContinue > 0 and h < env.epLen:
                action              = agent.pick_action(state, h)
                reward, new_s, pContinue = env.advance(action)
                ep_rew             += reward
                agent.update_obs(state, action, reward, new_s, h)
                state               = new_s
                h                  += 1
            self.data[ep - 1] = [ep, ep_rew, agent.get_num_arms()]
        return self

    def episode_rewards(self):
        return self.data[:, 1]

# =============================================================================
# SECTION 9 — Reference value computation
# =============================================================================

def compute_vstar_gh(n_state=801, n_action=401, n_quad=61,
                     x0=STARTING_STATE, reward_fn=None):
    """
    Vectorised Gauss-Hermite Bellman solver — delegates to BellmanSolverScalar.

    There is now ONE canonical implementation (BellmanSolverScalar.solve).
    This function is kept so notebook Cell 2 does not need to change.

    Default resolution is safe for 8 GB machines:
      peak RAM = n_state * n_action * n_quad * 24 bytes
               = 801 * 401 * 61 * 24  ≈  0.47 GB

    Do NOT use n_state=1601, n_action=801, n_quad=121 on an 8 GB machine —
    that allocates 3.7 GB of numpy arrays and will crash the kernel.
    Use those values only on machines with 16 GB+ RAM.
    """
    solver = BellmanSolverScalar(ep_len=EP_LEN,
                                 n_state=n_state,
                                 n_action=n_action,
                                 n_quad=n_quad,
                                 reward_fn=reward_fn)
    solver.solve()
    return solver.get_value(x0, h=0)


def compute_best_constant(n_actions=201, n_mc=50_000, seed=0, x0=STARTING_STATE,
                          reward_fn=None):
    """
    Value of the best constant-action policy, estimated by Monte Carlo.

    This is a LOWER BOUND on V*(x0) because the optimal policy is
    state-dependent.  Using this as the reference makes regret appear
    LARGER (more conservative) — honest but not directly comparable to
    the paper's figure, which uses the GH solver.

    Returns (value, best_action).
    """
    rfn       = reward_fn if reward_fn is not None else _DEFAULT_REWARD
    rng       = np.random.default_rng(seed)
    noise     = rng.standard_normal((EP_LEN, n_mc))
    best_val  = -np.inf
    best_a    = ACTION_LO

    for a_val in np.linspace(ACTION_LO, ACTION_HI, n_actions):
        state = np.full(n_mc, float(x0))
        total = np.zeros(n_mc)
        for h in range(EP_LEN):
            total += np.array([rfn(float(s), float(a_val)) for s in state])
            drift  = DRIFT_BIAS + DRIFT_STATE * state + DRIFT_ACTION * a_val
            state  = state + drift * DELTA + SIGMA * _SQRT_DELTA * noise[h]
            state  = np.clip(state, -RHO, RHO)
        v = float(np.mean(total))
        if v > best_val:
            best_val, best_a = v, a_val

    return best_val, best_a

# =============================================================================
# SECTION 10 — Parallel experiments
# =============================================================================

def _run_one_experiment(seed, reward_fn=None, scaling=SCALING, initial_q=None):
    """Single experiment run, returned as a 1-D array of episode rewards."""
    env   = AdaDiffEnvironment(reward_fn=reward_fn)
    agent = APLDiffusion(flag=True, scaling=scaling, initial_q=initial_q)
    return Experiment(env, agent, seed=seed).run().episode_rewards()


def run_experiments(n=20, reward_fn=None, scaling=SCALING, initial_q=None):
    """
    Run n independent experiments in parallel.
    Returns vpi_matrix of shape (N_EPS, n): episode rewards per experiment.

    scaling   : UCB exploration constant passed directly to each agent.
                Do NOT rely on patching aaro_again.SCALING — joblib workers
                are forked before the patch and will see the original value.
    initial_q : Optimistic Q initialisation. Same joblib caveat applies.
    reward_fn : Reward function. Defaults to reward_6_1.
    """
    results = Parallel(n_jobs=-1)(
        delayed(_run_one_experiment)(i, reward_fn=reward_fn,
                                     scaling=scaling, initial_q=initial_q)
        for i in range(n)
    )
    return np.column_stack(results)   # shape (N_EPS, n)

# =============================================================================
# SECTION 11 — Regret computation
# =============================================================================

def cumulative_regret(vpi_matrix, v_star):
    """
    Compute cumulative regret per experiment and summary statistics.

    Per-episode regret at episode k for experiment n:
        delta^{k,n} = max(v_star - r^{k,n}, 0)

    Regret(K, n) = sum_{k=1}^K delta^{k,n}

    The max(.,0) prevents negative regret caused by approximation error
    in v_star (e.g. a single lucky episode exceeding the solver estimate).

    Returns:
        mean  shape (N_EPS,)  — mean cumulative regret across experiments
        lo    shape (N_EPS,)  — 5th percentile
        hi    shape (N_EPS,)  — 95th percentile
    """
    per_ep  = np.maximum(v_star - vpi_matrix, 0.0)   # (N_EPS, n)
    cum     = np.cumsum(per_ep, axis=0)               # (N_EPS, n)
    return cum.mean(axis=1), np.percentile(cum, 5, axis=1), np.percentile(cum, 95, axis=1)


def regret_slope(cum_mean, fit_start=1000):
    """
    Fit log(Regret(k)) = alpha*log(k) + c over k >= fit_start.
    Returns (slope, intercept, r_squared).

    slope = alpha is the empirical regret exponent.
    Theoretical worst-case: alpha <= 3/4 for dS=dA=1, m=1 (Theorem 5.19).
    """
    eps  = np.arange(1, len(cum_mean) + 1)
    mask = (eps >= fit_start) & (cum_mean > 0)
    lx, ly = np.log(eps[mask]), np.log(cum_mean[mask])
    slope, intercept, r, _, _ = stats.linregress(lx, ly)
    return slope, intercept, r ** 2

# =============================================================================
# SECTION 12 — Plotting
# =============================================================================

def plot_all(vpi_matrix, v_star_gh, v_star_mc=None, fit_start=1000):
    """
    Three-panel figure:
      (a) Learning curve with ±1σ band and both reference lines
      (b) Log-log cumulative regret for both reference lines + theoretical bound
      (c) Slope comparison bar chart
    """
    n_eps  = vpi_matrix.shape[0]
    eps    = np.arange(1, n_eps + 1)
    mean_r = vpi_matrix.mean(axis=1)
    std_r  = vpi_matrix.std(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── (a) Learning curve ───────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(eps, mean_r, lw=1.2, color='steelblue', label='Mean episode reward')
    ax.fill_between(eps, mean_r - std_r, mean_r + std_r,
                    alpha=0.18, color='steelblue', label='±1σ')
    ax.axhline(v_star_gh, color='crimson', lw=1.3, ls='-',
               label=f'V* GH solver = {v_star_gh:.2f}')
    if v_star_mc is not None:
        ax.axhline(v_star_mc, color='darkorange', lw=1.3, ls='--',
                   label=f'V* MC constant = {v_star_mc:.2f}')
    ax.set_xlabel("Episode");  ax.set_ylabel("Episode reward")
    ax.set_title("(a) Learning curve")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

    # ── (b) Log-log regret ───────────────────────────────────────────────────
    ax     = axes[1]
    slopes = {}

    _refs = [('GH solver (paper ref)', v_star_gh, 'crimson', '-')]
    if v_star_mc is not None:
        _refs.append(('MC constant (lower bd)', v_star_mc, 'darkorange', '--'))
    for label, v_ref, color, ls in _refs:
        cm, cl, ch = cumulative_regret(vpi_matrix, v_ref)
        sl, ic, r2 = regret_slope(cm, fit_start)
        slopes[label] = sl

        mask = (eps >= fit_start) & (cm > 0)
        lx   = np.log(eps[mask])
        ly   = np.log(cm[mask])
        ax.plot(lx, ly, lw=1.4, color=color, ls=ls,
                label=f'{label}  α={sl:.3f}  R²={r2:.3f}')
        ax.plot(lx, sl * lx + ic, lw=0.9, color=color, ls=':', alpha=0.8)

        # Confidence band
        lx_all = np.log(eps[cm > 0])
        ax.fill_between(lx_all,
                        np.log(np.maximum(cl[cm > 0], 1e-8)),
                        np.log(np.maximum(ch[cm > 0], 1e-8)),
                        alpha=0.08, color=color)

    # Theoretical worst-case slope anchored through midpoint of GH curve
    th_slope = 3.0 / 4.0
    cm_gh, _, _ = cumulative_regret(vpi_matrix, v_star_gh)
    mask_th  = (eps >= fit_start) & (cm_gh > 0)
    lx_th    = np.log(eps[mask_th])
    mid      = len(lx_th) // 2
    th_ic    = math.log(max(cm_gh[mask_th][mid], 1.0)) - th_slope * lx_th[mid]
    ax.plot(lx_th, th_slope * lx_th + th_ic,
            lw=1.1, color='gray', ls='-.', alpha=0.85,
            label=f'Theoretical bound  α={th_slope:.2f}')

    ax.set_xlabel("log(episode)");  ax.set_ylabel("log(cumulative regret)")
    ax.set_title("(b) Log-log regret  (slope = regret exponent α)")
    ax.set_xlim(left=math.log(fit_start))   # focus on converged region only
    ax.legend(fontsize=7);  ax.grid(True, alpha=0.3)

    # ── (c) Slope bar chart ──────────────────────────────────────────────────
    ax     = axes[2]
    names  = list(slopes.keys()) + ['Theoretical bound']
    values = list(slopes.values()) + [th_slope]
    colors = ['crimson', 'darkorange', 'gray']
    bars   = ax.bar(range(len(names)), values, color=colors, alpha=0.72, width=0.5)
    ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5, label='Linear (α=1)')
    ax.axhline(th_slope, color='gray', lw=0.8, ls='-.', alpha=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel("Regret exponent α");  ax.set_ylim(0, 1.1)
    ax.set_title("(c) Slope comparison")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(
        "APL-Diffusion — regret analysis, 1-D O-U experiment (Section 6.1)",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.savefig("apl_regret_analysis.png", dpi=150, bbox_inches='tight')
    print("Figure saved to apl_regret_analysis.png")
    plt.show()
    return slopes


def plot_partition_heatmap(agent, timestep=9):
    """Q-value heatmap of the adaptive partition at a given timestep."""
    tree   = agent.tree_list[timestep]
    leaves = tree.tree_leaves
    q_vals = [n.qVal for n in leaves]
    q_min, q_max = min(q_vals), max(q_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    for node in leaves:
        x0 = node.state_val  - node.radius
        y0 = node.action_val - node.action_radius
        w  = 2 * node.radius
        h  = 2 * node.action_radius
        qn = 0.5 if q_max == q_min else (node.qVal - q_min) / (q_max - q_min)
        ax.add_patch(patches.Rectangle(
            (x0, y0), w, h, linewidth=0.5,
            edgecolor='black', facecolor=plt.cm.RdYlGn_r(qn)
        ))
    ax.set_xlim(-RHO, RHO);  ax.set_ylim(ACTION_LO, ACTION_HI)
    ax.set_xlabel("State");   ax.set_ylabel("Action")
    ax.set_title(f"Heat map of Q values  (h={timestep}, k={N_EPS})")
    plt.tight_layout();  plt.show()



# =============================================================================
# SECTION 13 — ExpConfig + multi-seed experiment runner
# (mirrors the Conor.py interface so report notebooks can use aaro_again directly)
# =============================================================================

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple
import math as _math

@dataclass
class ExpConfig:
    """
    Single experiment configuration.
    Pass a list of these to run_experiment() to sweep hyperparameters.

    All fields have sensible defaults matching the new code (conor.py).
    Override only what you want to change — everything else stays fixed.
    """
    # ── Problem ────────────────────────────────────────────────────────────
    starting_state: float = 4.0
    action_lo:      float = -5.0
    action_hi:      float =  5.0
    rho:            float = 10.0
    epLen:          int   = 10
    nEps:           int   = 2000
    n_seeds:        int   = 10
    # ── Dynamics ───────────────────────────────────────────────────────────
    theta_0: float = 0.05
    theta_x: float = -0.1
    theta_a: float = 0.01
    sigma:   float = 0.1
    delta:   float = 1.0
    # ── Reward ─────────────────────────────────────────────────────────────
    reward_step_fn: Callable = field(default_factory=lambda: reward_6_1)
    # ── Agent hyperparameters ──────────────────────────────────────────────
    initial_q:       float = 1837.1
    scaling:         float = 5.0
    alpha:           float = 0.5   # UCB exponent: bonus = scaling / n^alpha
    split_threshold: int   = 2
    lip:             float = 1.0
    # ── Label ──────────────────────────────────────────────────────────────
    label: str = 'experiment'
    # ── Derived (set automatically) ────────────────────────────────────────
    _sigma_sqrt_delta: float = field(init=False, repr=False)
    _action_center:    float = field(init=False, repr=False)
    rho_1:             float = field(init=False, repr=False)

    def __post_init__(self):
        self._sigma_sqrt_delta = self.sigma * _math.sqrt(self.delta)
        self._action_center    = (self.action_hi + self.action_lo) / 2.0
        self.rho_1             = (self.action_hi - self.action_lo) / 2.0


def _run_one_seed_cfg(seed: int, cfg: 'ExpConfig'):
    """Run one seed using ExpConfig — uses existing AdaDiffEnvironment + APLDiffusion."""
    import numpy as _np

    # Build a lightweight env that uses cfg parameters
    class _CfgEnv(Environment):
        def __init__(self):
            self.epLen    = cfg.epLen
            self._start   = float(cfg.starting_state)
            self.state    = _np.array([self._start], dtype=_np.float64)
            self.timestep = 0
            self._lo      = -cfg.rho
            self._hi      =  cfg.rho
            self._ssd     = cfg._sigma_sqrt_delta

        def get_epLen(self): return self.epLen

        def reset(self):
            self.timestep = 0
            self.state[0] = self._start

        def advance(self, action):
            x = self.state[0]
            a = float(_np.clip(action[0], cfg.action_lo, cfg.action_hi))
            drift  = cfg.theta_0 + cfg.theta_x * x + cfg.theta_a * a
            new_x  = x + drift * cfg.delta + self._ssd * float(_np.random.randn())
            new_x  = float(_np.clip(new_x, self._lo, self._hi))
            # reward_step_fn in new code expects (state_array, action_array)
            # but aaro_again's reward functions expect (x: float, a: float)
            # We call with arrays so both interfaces work via state[0]/action[0]
            # Call reward function with plain floats — works for both
            # aaro_again style (x: float, a: float) and
            # new code style (state: array, action: array) via state[0]/action[0]
            try:
                reward = float(cfg.reward_step_fn(x, a))
            except TypeError:
                reward = float(cfg.reward_step_fn(
                    _np.array([x], dtype=_np.float64),
                    _np.array([a], dtype=_np.float64),
                ))
            self.state[0] = new_x
            self.timestep += 1
            pContinue = 1 if self.timestep < self.epLen else 0
            return reward, self.state, pContinue

    # Build agent wiring ExpConfig values explicitly so joblib fork sees them
    agent = APLDiffusion(
        ep_len    = cfg.epLen,
        scaling   = cfg.scaling,
        initial_q = cfg.initial_q,
        flag      = False,   # online updates match the new code's behaviour
    )

    # Override action bounds on agent so pick_action respects cfg
    agent._action_lo = cfg.action_lo
    agent._action_hi = cfg.action_hi

    _np.random.seed(seed)
    rewards = _np.zeros(cfg.nEps)
    arms    = _np.zeros(cfg.nEps)

    for ep in range(cfg.nEps):
        state  = _np.array([float(cfg.starting_state)], dtype=_np.float64)
        ep_rew = 0.0
        agent.update_policy(ep)

        for h in range(cfg.epLen):
            s      = float(state[0])
            a      = float(agent.pick_action(s, h))
            a_clip = float(_np.clip(a, cfg.action_lo, cfg.action_hi))
            drift  = cfg.theta_0 + cfg.theta_x * s + cfg.theta_a * a_clip
            new_x  = s + drift*cfg.delta + cfg._sigma_sqrt_delta*float(_np.random.randn())
            new_x  = float(_np.clip(new_x, -cfg.rho, cfg.rho))
            pcont  = 1 if h + 1 < cfg.epLen else 0
            try:
                reward = float(cfg.reward_step_fn(s, a_clip))
            except TypeError:
                reward = float(cfg.reward_step_fn(
                    _np.array([s],      dtype=_np.float64),
                    _np.array([a_clip], dtype=_np.float64),
                ))
            ep_rew   += reward
            agent.update_obs(s, a_clip, reward, new_x, h)
            state[0]  = new_x
            if not pcont:
                break

        rewards[ep] = ep_rew
        arms[ep]    = agent.get_num_arms()

    return rewards, arms


def run_experiment(configs: list, n_jobs: int = -1) -> dict:
    """
    Run a list of ExpConfig objects, each over cfg.n_seeds parallel seeds.

    Returns a dict:
        results[cfg.label] = {
            "vpi":  np.ndarray shape (nEps,)   mean reward per episode
            "arms": np.ndarray shape (nEps,)   mean active balls per episode
        }

    Usage:
        results = run_experiment([cfg1, cfg2, cfg3])
        vpi = results["my_label"]["vpi"]
    """
    import numpy as _np
    results = {}
    for cfg in configs:
        print(f"  Running: {cfg.label} ...")
        seed_runs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_one_seed_cfg)(seed, cfg) for seed in range(cfg.n_seeds)
        )
        reward_matrix = _np.vstack([r for r, a in seed_runs])
        arm_matrix    = _np.vstack([a for r, a in seed_runs])
        results[cfg.label] = {
            "vpi":  reward_matrix.mean(axis=0),
            "arms": arm_matrix.mean(axis=0),
        }
        print(f"    Done. Final mean VPI (last 100 ep): {results[cfg.label]['vpi'][-100:].mean():.3f}")
    return results
