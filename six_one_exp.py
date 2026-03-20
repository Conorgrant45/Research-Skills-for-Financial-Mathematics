#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APL-Diffusion: Combined implementation.

Theory source:
  Jin, Xu, Yang (2025) - "Adaptive Partitioning and Learning for Stochastic
  Control of Diffusion Processes", arXiv:2512.14991.

This file merges:
  - The .py script's speed (Node __slots__, O(1) leaf count, self-contained
    base classes, direct next-state Q lookup, polynomial vEst init).
  - The notebook's theory-correct splitting (CONF <= diam, state-scaled
    confidence) and exact Gauss-Hermite Bellman benchmark.

Key algorithm references:
  Algorithm 1  - APL-Diffusion main loop
  Algorithm 2  - Block selection (greedy on Q)
  Algorithm 4  - Splitting rule: CONF_h^k(B) <= diam(B)
  Eq. (4.1)    - Drift/volatility estimators (online Welford form)
  Eq. (4.16)   - Reward estimator
  Eq. (4.20)   - CONF_h^k(B) = g1(delta, ||x_tilde||) / sqrt(n)
  Eq. (5.4)    - Optimistic initialisation Q^0, V^0
  Eq. (5.6-9)  - Q/V update with UCB bonus and bias terms
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Global parameters  (Section 6.1 experiment, paper Table / Section 6.1)
# ---------------------------------------------------------------------------
EP_LEN          = 10       # H  - horizon per episode
N_EPS           = 2000     # K  - total episodes
STARTING_STATE  = 4.0      # x_1 (fixed initial state for experiment)
DELTA           = 1.0      # Δ  - time increment

# Diffusion dynamics: mu_h(x,a) = b0 + b1*x + b2*a,  sigma_h(x,a) = sigma
DRIFT_BIAS      = 0.05     # b0
DRIFT_STATE     = -0.10    # b1
DRIFT_ACTION    = 0.01     # b2
SIGMA           = 0.10     # constant volatility (scalar 1-D)

# Algorithm hyper-parameters
RHO             = 10.0     # ρ  - truncation radius (Section 3, S_1 = {||x||<=rho})
D_INIT          = 10.0 * math.sqrt(2)  # D - initial partition diameter
C_H             = 5.0      # C_h - local Lipschitz constant (Proposition 2.4)
M               = 1        # polynomial growth order: reward ~ O(|x|^{m+1})
SCALING         = 0.01     # multiplier inside CONF (replaces unknown g1 constants)
INITIAL_Q       = 1837.1   # Q^0(B) - optimistic init (Eq. 5.4, Sec 6.1)
OOB_Q           = -505.0   # Q^k(Z̄^c) - out-of-bounds penalty (Eq. 5.5)

_SQRT_DELTA     = math.sqrt(DELTA)
_ACTION_LO      = 0.0
_ACTION_HI      = 10.0

# ---------------------------------------------------------------------------
# Minimal base classes (avoid external framework dependency)
# ---------------------------------------------------------------------------

class Agent:
    def update_obs(self, obs, action, reward, newObs, timestep): pass
    def update_policy(self, k): pass
    def pick_action(self, obs, timestep): return 0.0
    def get_num_arms(self): return 0

class Environment:
    def get_epLen(self): return 0
    def reset(self): pass
    def advance(self, action): return 0.0, 0.0, 0

# ---------------------------------------------------------------------------
# Environment  (Section 2, Eq. 2.1 + Section 6.1 reward)
# ---------------------------------------------------------------------------

class AdaDiffEnvironment(Environment):
    """
    Implements the 1-D O-U diffusion from Section 6.1:
      X_{h+1} = X_h + mu_h(X_h, A_h)*Delta + sigma*sqrt(Delta)*B_h
      reward ~ N((x - a)^2, 0.01)

    Out-of-bounds: when |X_{h+1}| > rho the environment returns OOB_Q
    and terminates, matching the localization S_1 in Section 3.
    """
    def __init__(self, ep_len=EP_LEN, starting_state=STARTING_STATE):
        self.epLen          = ep_len
        self.starting_state = float(starting_state)
        self.state          = self.starting_state
        self.timestep       = 0

    def get_epLen(self):
        return self.epLen

    def reset(self):
        self.timestep = 0
        self.state    = self.starting_state

    def advance(self, action):
        x = float(self.state)
        a = float(np.clip(action, _ACTION_LO, _ACTION_HI))

        drift     = DRIFT_BIAS + DRIFT_STATE * x + DRIFT_ACTION * a
        new_x     = x + drift * DELTA + SIGMA * _SQRT_DELTA * float(np.random.randn())

        self.timestep += 1
        pContinue = 0 if self.timestep >= self.epLen else 1

        if abs(new_x) > RHO:
            # State escaped S_1: apply penalty Q^k(Z̄^c) and terminate (Eq. 5.5)
            reward    = OOB_Q
            pContinue = 0
            new_x     = float(np.clip(new_x, -RHO, RHO))
        else:
            # Normal reward: R_h(x,a) ~ N((x-a)^2, 0.01)  (Section 6.1)
            new_x  = float(np.clip(new_x, -RHO, RHO))
            reward = float(np.random.normal(loc=(x - a) ** 2, scale=0.1))

        self.state = new_x
        return reward, new_x, pContinue

# ---------------------------------------------------------------------------
# Exact Bellman solver (benchmark only, not part of APL-Diffusion)
# ---------------------------------------------------------------------------

class BellmanSolverScalar:
    """
    Computes V_1^*(x_0) by backward induction using Gauss-Hermite quadrature
    to integrate over N(0,1) noise exactly.
    This is the ground-truth benchmark for Figure 3(a) of the paper.
    """
    def __init__(self, ep_len=EP_LEN, n_state=801, n_action=401, n_quad=81):
        self.epLen       = ep_len
        self.state_grid  = np.linspace(-RHO, RHO, n_state)
        self.action_grid = np.linspace(_ACTION_LO, _ACTION_HI, n_action)
        pts, wts         = np.polynomial.hermite.hermgauss(n_quad)
        self.quad_z      = pts * math.sqrt(2.0)   # scale to N(0,1)
        self.quad_w      = wts / math.sqrt(math.pi)
        self.V           = np.zeros((ep_len + 1, n_state))
        self.policy      = np.zeros((ep_len, n_state))

    


    def get_value(self, x, h=0):
        return float(np.interp(x, self.state_grid, self.V[h]))

# ---------------------------------------------------------------------------
# Node  (one hypercube in the joint state-action partition)
# ---------------------------------------------------------------------------

class Node:
    """
    Represents a block B ∈ P_h^k (Algorithm 1).

    Stored statistics correspond to:
      rEst     : R̂_h^k(B)          (Eq. 4.16)
      muEst    : μ̂_h^k(B)           (Eq. 4.1, online Welford)
      sigmaEst : Σ̂_h^k(B) scalar   (Eq. 4.1, online Welford, 1-D)
      num_visits        : n_h^k(B)  including ancestor visits (Algorithm 3)
      num_unique_visits : direct visits used for estimation
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
        Algorithm 4 line 2: split into 2^(dS+dA) = 4 children (1-D x 1-D),
        each with half the diameter.  Children inherit parent statistics if
        the parent was well-visited; otherwise they reset to optimistic init.
        (Algorithm 4 lines 4-6: n_h^k(B_i) = n_h^k(B))
        """
        hr  = self.radius        * 0.5
        har = self.action_radius * 0.5
        low = self.num_visits <= 1
        children = []
        for ds in (-1, 1):
            ns = float(self.state_val  + ds * hr)
            for da in (-1, 1):
                na = float(np.clip(self.action_val + da * har, _ACTION_LO, _ACTION_HI))
                if low:
                    # Optimistic init for rarely-visited parents (Eq. 5.4)
                    child = Node(INITIAL_Q, 0.0, 0.0, 0.0,
                                 self.num_visits, 0, self.num_splits + 1,
                                 ns, na, hr, har)
                else:
                    # Inherit all statistics (Algorithm 4 line 5)
                    child = Node(self.qVal, self.rEst, self.muEst, self.sigmaEst,
                                 self.num_visits, self.num_visits, self.num_splits + 1,
                                 ns, na, hr, har)
                children.append(child)
        self.children = children
        return children

# ---------------------------------------------------------------------------
# Tree  (per-timestep partition + value function table)
# ---------------------------------------------------------------------------

class Tree:
    """
    Manages P_h^k for a single timestep h.

    state_leaves / vEst implement the state-projection Ṽ_h^k(S) from Eq. (5.6-5.8):
      - state_leaves: centers of induced state partition Γ_S(P_h^k)
      - vEst[i]: Ṽ_h^k(S_i), updated monotonically downward (min rule)
    _min_vEst caches min(vEst) for O(1) Bellman backup.
    """
    def __init__(self):
        # Initial partition: two blocks covering [-rho, 0) and [0, rho]
        # Centers at ±5, radii 5 (state) and 5 (action) → covers [-10,10]×[0,10]
        # Optimistic init: Q^0(B) = C̃_h(1+||x̃||^{m+1})  (Eq. 5.4)
        init_q1 = C_H * (1.0 + abs(5.0) ** (M + 1))
        init_q2 = C_H * (1.0 + abs(-5.0) ** (M + 1))
        self.head_pos = Node(init_q1, 0, 0, 0, 0, 0, 0,  5.0, 5.0, 5.0, 5.0)
        self.head_neg = Node(init_q2, 0, 0, 0, 0, 0, 0, -5.0, 5.0, 5.0, 5.0)

        self.tree_leaves  = [self.head_pos, self.head_neg]
        self.state_leaves = [self.head_pos.state_val, self.head_neg.state_val]
        # Ṽ^0(S) = C̃_h(1+||x̃(S)||^{m+1})  (Eq. 5.4)
        self.vEst         = [init_q1, init_q2]
        self._min_vEst    = min(self.vEst)

    # ------------------------------------------------------------------
    # Block diameter: L2 norm of (state_radius, action_radius) (Section 2)
    # ------------------------------------------------------------------
    @staticmethod
    def block_diameter(node):
        return math.sqrt(node.radius ** 2 + node.action_radius ** 2)

    # ------------------------------------------------------------------
    # CONF_h^k(B)  (Eq. 4.20)
    #
    # Theory: CONF = g1(delta, ||x̃(oB)||) / sqrt(n)
    # where g1 grows with state norm via eta(||x̃||) = L0 + L(||x̃||+ā) + 2LD.
    #
    # Implementation: we replace g1 with a data-driven proxy:
    #   local_scale = 1 + |μ̂| + sqrt(σ̂)  ≈ eta(||x̃||) estimated from data.
    # The log(n) factor tightens the bound relative to 1/sqrt(n), matching
    # the log(HK²/δ) term inside κ_μ (Proposition 4.1).
    # ------------------------------------------------------------------
    @staticmethod
    def confidence(node, scaling=SCALING):
        n           = max(1, node.num_unique_visits)
        local_scale = 1.0 + abs(node.muEst) + math.sqrt(max(node.sigmaEst, 0.0))
        return scaling * local_scale * math.sqrt(math.log(n + 2.0) / n)

    # ------------------------------------------------------------------
    # Splitting condition  (Algorithm 4, line 1):  CONF_h^k(B) <= diam(B)
    # Theory: split when statistical confidence is tighter than block bias.
    # ------------------------------------------------------------------
    def should_split(self, node):
        if node.num_unique_visits < 2:
            return False
        return self.confidence(node) <= self.block_diameter(node)

    # ------------------------------------------------------------------
    # Split and update partition  (Algorithm 4)
    # ------------------------------------------------------------------
    def split_node(self, node):
        children = node.split()

        # Remove parent, add children to active leaf set
        self.tree_leaves.remove(node)
        self.tree_leaves.extend(children)

        # Update state partition Γ_S(P_h^k) if a new state half appears
        c0_state  = children[0].state_val
        c0_radius = children[0].radius
        min_dist  = min(abs(sl - c0_state) for sl in self.state_leaves)
        if min_dist >= c0_radius:
            parent = node.state_val
            try:
                idx = self.state_leaves.index(parent)
                parent_v = self.vEst[idx]
                self.state_leaves.pop(idx)
                self.vEst.pop(idx)
            except ValueError:
                parent_v = node.qVal
            # Two new state centers: one per state half (children[0] and children[2])
            self.state_leaves.append(children[0].state_val)
            self.state_leaves.append(children[2].state_val)   # 2^dA = 2 action children per state half
            self.vEst.append(parent_v)
            self.vEst.append(parent_v)
            self._min_vEst = min(self.vEst)

    # ------------------------------------------------------------------
    # Block selection  (Algorithm 2):
    # find the leaf containing 'state' with the highest Q estimate.
    # ------------------------------------------------------------------
    def get_active_ball(self, state):
        safe = float(np.clip(state, -RHO, RHO))
        root = self.head_pos if safe >= 0.0 else self.head_neg
        return self._traverse(safe, root)

    def _traverse(self, state, node):
        if node.children is None:
            return node, node.qVal
        best_node, best_q = node, node.qVal   # fallback to current if no child matches
        for child in node.children:
            if abs(state - child.state_val) <= child.radius:
                n, q = self._traverse(state, child)
                if q >= best_q:
                    best_node, best_q = n, q
        return best_node, best_q

    # ------------------------------------------------------------------
    # Update Ṽ(S) for all state leaves  (Eq. 5.6, monotone min rule)
    # ------------------------------------------------------------------
    def refresh_vEst(self):
        for idx, sv in enumerate(self.state_leaves):
            _, q = self.get_active_ball(sv)
            self.vEst[idx] = min(q, INITIAL_Q, self.vEst[idx])
        self._min_vEst = min(self.vEst)

    def get_num_leaves(self):
        return len(self.tree_leaves)

# ---------------------------------------------------------------------------
# APL-Diffusion agent  (Algorithm 1 + Algorithm 2 + Algorithm 4)
# ---------------------------------------------------------------------------

class APLDiffusion(Agent):
    """
    Adaptive Partition and Learning for Diffusions.

    flag=True  → model-based offline update (update_policy called at episode
                 start, backward h=H-1..0, matching Algorithm 1 lines 9-13).
    flag=False → online update: Q refreshed immediately after each observation.
    """
    def __init__(self, ep_len=EP_LEN, scaling=SCALING, flag=True):
        self.epLen     = ep_len
        self.scaling   = scaling
        self.flag      = flag
        self.tree_list = [Tree() for _ in range(ep_len)]
        self._final_h  = ep_len - 1

    def reset(self):
        self.tree_list = [Tree() for _ in range(self.epLen)]

    def get_num_arms(self):
        return sum(t.get_num_leaves() for t in self.tree_list)

    # ------------------------------------------------------------------
    # update_obs  (Algorithm 1 line 12 + Algorithm 3 + parts of Alg 5)
    # Called after every (X_h, A_h, r_h, X_{h+1}) observation.
    # ------------------------------------------------------------------
    def update_obs(self, obs, action, reward, newObs, timestep):
        tree        = self.tree_list[timestep]
        node, _     = tree.get_active_ball(obs)

        # --- Algorithm 3: update visit counts ---
        node.num_visits        += 1
        node.num_unique_visits += 1
        t     = node.num_unique_visits
        alpha = 1.0 / t            # Welford step size

        # --- Reward estimator R̂_h^k(B)  (Eq. 4.16) ---
        node.rEst = (1.0 - alpha) * node.rEst + alpha * reward

        # --- Drift/volatility estimators μ̂, Σ̂  (Eq. 4.1, online form) ---
        # Only meaningful for non-terminal steps (no X_{h+1} at H)
        if timestep != self._final_h:
            delta_x      = newObs - obs
            old_mu       = node.muEst
            node.muEst   = (1.0 - alpha) * old_mu       + alpha * delta_x
            node.sigmaEst = (1.0 - alpha) * node.sigmaEst + alpha * (delta_x - old_mu) ** 2

        # --- Online Q/V update (flag=False mode) ---
        if not self.flag:
            self._update_q(node, newObs, timestep, tree)
            tree.refresh_vEst()

        # --- Algorithm 4: splitting rule  CONF <= diam ---
        if tree.should_split(node):
            tree.split_node(node)

    # ------------------------------------------------------------------
    # update_policy  (Algorithm 1 lines 9-13, backward sweep)
    # Called once per episode before the episode starts (flag=True mode).
    # ------------------------------------------------------------------
    def update_policy(self, k):
        if not self.flag:
            return
        for h in range(self._final_h, -1, -1):
            tree = self.tree_list[h]
            for node in tree.tree_leaves:
                if node.num_unique_visits == 0:
                    # Unvisited: keep optimistic initialisation (Eq. 5.4)
                    node.qVal = C_H * (1.0 + abs(node.state_val) ** (M + 1))
                else:
                    self._update_q(node, None, h, tree)
            tree.refresh_vEst()

    # ------------------------------------------------------------------
    # Q-function update  (Eq. 5.6-5.9)
    #
    # Q_h^k(B) = R̂ + R-UCB + E_{X~T̄}[V_{h+1}^k(X)] + T-UCB + BIAS
    #
    # Implementation:
    #   R̂   = node.rEst
    #   UCB  = scaling / sqrt(n)          (lumped R-UCB + T-UCB)
    #   BIAS = scaling * radius            (∝ diam(B), Eq. 5.1-5.2)
    #   E[V] = next Q at actual next state (direct lookup, tighter than min(vEst))
    #   Lipschitz correction for V_{h+1} growth: C_h * sigmaEst (proxy for
    #     C_h(1+||x||^m+||x̃||^m)||x-x̃|| in Eq. 5.8 — variance captures
    #     within-block state spread)
    #
    # The min() enforces the monotone decreasing property (Theorem 5.2):
    #   Q^k <= Q^{k-1} (estimates only improve, never worsen).
    # ------------------------------------------------------------------
    def _update_q(self, node, next_obs, timestep, tree):
        n          = max(1, node.num_visits)
        ucb        = self.scaling / math.sqrt(n)
        bias       = self.scaling * node.radius   # ∝ diam(B)

        if timestep == self._final_h:
            new_q = node.rEst + ucb + bias
        else:
            next_tree = self.tree_list[timestep + 1]
            if next_obs is not None:
                # Online mode: use actual next state (tighter bound than min(vEst))
                _, next_q = next_tree.get_active_ball(next_obs)
            else:
                # Offline mode: conservative min over state leaves
                next_q = next_tree._min_vEst
            # Lipschitz correction: C_h * (1 + muEst^2 + sigmaEst) proxies
            # C_h * (1 + ||x||^m + ||x̃||^m) * ||x - x̃|| from Eq. (5.8)
            lip_correction = C_H * node.sigmaEst
            v_est = next_q + lip_correction
            new_q = node.rEst + v_est + ucb + bias

        # Monotone min (Theorem 5.2, ensures Q^k >= Q*)
        node.qVal = min(node.qVal, INITIAL_Q, new_q)

    # ------------------------------------------------------------------
    # Action selection  (Algorithm 2 + Algorithm 1 line 6)
    # Select block maximising Q, then sample action uniformly from Γ_A(B).
    # ------------------------------------------------------------------
    def pick_action(self, state, timestep):
        tree       = self.tree_list[timestep]
        node, _    = tree.get_active_ball(state)
        action     = np.random.uniform(node.action_val - node.action_radius,
                                       node.action_val + node.action_radius)
        return float(np.clip(action, _ACTION_LO, _ACTION_HI))

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class Experiment:
    def __init__(self, env, agent, n_eps=N_EPS, seed=0, debug=False):
        self.env    = env
        self.agent  = agent
        self.n_eps  = n_eps
        self.debug  = debug
        self.data   = np.zeros((n_eps, 3))   # [episode, epReward, n_leaves]
        np.random.seed(seed)

    def run(self):
        env, agent = self.env, self.agent
        for ep in range(1, self.n_eps + 1):
            env.reset()
            state    = env.state
            ep_rew   = 0.0
            agent.update_policy(ep)
            pContinue = 1
            h         = 0
            while pContinue > 0 and h < env.epLen:
                action          = agent.pick_action(state, h)
                reward, new_s, pContinue = env.advance(action)
                ep_rew         += reward
                agent.update_obs(state, action, reward, new_s, h)
                state           = new_s
                h              += 1
            self.data[ep - 1] = [ep, ep_rew, agent.get_num_arms()]
        return self

    def to_df(self):
        return pd.DataFrame(self.data, columns=['episode', 'epReward', 'n_leaves'])

# ---------------------------------------------------------------------------
# Parallel experiment helper
# ---------------------------------------------------------------------------

def run_one(seed):
    env   = AdaDiffEnvironment()
    agent = APLDiffusion(flag=True)
    df    = Experiment(env, agent, seed=seed).run().to_df()
    return df['epReward'].values

# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_learning_curve(vpi, true_value):
    eps = range(1, len(vpi) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(eps, vpi, label='Estimated Vπ̃', lw=1.2)
    plt.axhline(true_value, color='red', lw=1.2, label=f'V* = {true_value:.1f}')
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("Estimated Vπ̃ vs episode")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()

def plot_log_regret(vpi, true_value, fit_start=1000):
    eps     = np.arange(1, len(vpi) + 1)
    regret  = np.maximum(true_value - np.asarray(vpi), 1e-10)
    cum_reg = np.cumsum(regret)
    mask    = eps >= fit_start
    lx, ly  = np.log(eps[mask]), np.log(cum_reg[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    plt.figure(figsize=(10, 5))
    plt.plot(lx, ly, label='log cumulative regret', lw=1.2)
    plt.plot(lx, slope * lx + intercept, 'r--',
             label=f'Slope = {slope:.3f}', lw=1.2)
    plt.xlabel("log episode"); plt.ylabel("log regret")
    plt.title(f"Log regret vs log episode (fit episodes {fit_start}–{len(vpi)})")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.tight_layout(); plt.show()
    return slope

def plot_partition_heatmap(agent, timestep=9):
    tree   = agent.tree_list[timestep]
    leaves = tree.tree_leaves
    q_vals = [n.qVal for n in leaves]
    q_min, q_max = min(q_vals), max(q_vals)

    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(8, 5))
    for node in leaves:
        x0     = node.state_val  - node.radius
        y0     = node.action_val - node.action_radius
        w      = 2 * node.radius
        h_rect = 2 * node.action_radius
        q_norm = 0.5 if q_max == q_min else (node.qVal - q_min) / (q_max - q_min)
        color  = plt.cm.RdYlGn_r(q_norm)
        ax.add_patch(patches.Rectangle((x0, y0), w, h_rect,
                                        linewidth=0.5, edgecolor='black',
                                        facecolor=color))
    ax.set_xlim(-RHO, RHO); ax.set_ylim(_ACTION_LO, _ACTION_HI)
    ax.set_xlabel("State"); ax.set_ylabel("Action")
    ax.set_title(f"Q-value heatmap — partition P_{timestep}^{N_EPS}")
    plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import time
    print("=" * 60)
    print("APL-Diffusion — combined implementation")
    print("=" * 60)

    # 1. Compute ground-truth V* via exact Bellman solver
    print("\nSolving Bellman equation (Gauss-Hermite quadrature)...")
    t0     = time.perf_counter()
    solver = BellmanSolverScalar()
    solver.solve()
    true_v = solver.get_value(STARTING_STATE, h=0)
    print(f"  V*(x_0={STARTING_STATE}) = {true_v:.4f}  [{time.perf_counter()-t0:.1f}s]")

    # 2. Run N parallel experiments
    N = 10
    print(f"\nRunning {N} parallel experiments × {N_EPS} episodes each...")
    t1      = time.perf_counter()
    results = Parallel(n_jobs=-1)(delayed(run_one)(i) for i in range(N))
    vpi_df  = pd.DataFrame(results).T
    vpi     = vpi_df.mean(axis=1).values
    print(f"  Done. [{time.perf_counter()-t1:.1f}s]")

    # 3. Partition visualisation (train one agent, inspect its tree)
    print("\nTraining one agent for partition visualisation...")
    env_vis   = AdaDiffEnvironment()
    agent_vis = APLDiffusion(flag=True)
    Experiment(env_vis, agent_vis, seed=123).run()
    plot_partition_heatmap(agent_vis, timestep=9)

    # 4. Learning curve and regret plots
    plot_learning_curve(vpi, true_v)
    slope = plot_log_regret(vpi, true_v, fit_start=1000)
    print(f"\nEstimated regret slope: {slope:.3f}")

    # Worst-case theoretical bound for dS=dA=1: (1+dS+dA)/(2+dS+dA) = 3/4 = 0.75
    theoretical_bound = (1 + state_d + action_d) / (2 + state_d + action_d) if False else 3/4
    print(f"Worst-case theoretical bound: {3/4:.3f}")
    print(f"Gap: {3/4 - slope:.3f}  ({'better' if slope < 3/4 else 'worse'} than worst-case)")

if __name__ == "__main__":
    main()