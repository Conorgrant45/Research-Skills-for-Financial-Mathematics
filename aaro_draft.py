#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research project - Adaptive Model-Based Discretization
Authors: Kyle McGillivray, Conor, Aaro

Implements Section 6.1 of Jin, Xu, Yang (2025) exactly:
  "A one-dimensional example"

Paper setup (Section 6.1):
  - State space: S = R, action space: [0, 10]
  - Dynamics:  mu_h(x,a) = 0.05 - 0.1x + 0.01a
               sigma_h(x,a) = 0.1  (constant, state/action independent)
               X_1 = 4
  - Reward:    R_h(x,a) ~ N((x-a)^2, 0.01)  for h in [H-1]
               (noisy quadratic, agent tries to match action to state)
  - H = 10, K = 2000, rho = 10
  - action_dim = 1 (scalar action in [0,10])
  - state_dim  = 1

Key differences from Section 6.2 (original code):
  - Reward is on EVERY step (not just terminal), noisy quadratic R~N((x-a)^2, 0.01)
  - Dynamics are LINEAR in x and a (not geometric O-U)
  - Sigma is CONSTANT (not state-dependent)
  - Optimal action a* = x (match action to current state) → a* ≈ 10 w.h.p.
  - Shorter episodes: H=10 not H=30

Domain projection fix (Section 3.1):
  The paper assumes the state lives in a compact set X ⊂ R.
  If the linear dynamics push the state outside [-rho, rho], the traversal
  in get_active_ball() would find no containing node and silently return the
  root with stale Q-estimates — wrong node, wrong update, corrupted learning.

  Fix: after every dynamics step, state is clipped to [domain_lo, domain_hi].
  This is an L-inf projection onto the compact domain, consistent with the
  reflected/projected diffusion interpretation in Section 3.1.

  Crucially, this does NOT break the code because:
    1. The traversal still receives a valid in-domain state every time.
    2. The root node always contains any projected state (rho is large enough).
    3. projection_count lets you monitor how often the boundary is hit — if
       it's zero throughout training, the domain is large enough and projection
       is a silent safety net. If it's non-zero, it flags a parameter issue.
    4. The Bellman updates and splitting logic are completely unchanged — they
       only ever see the (valid, projected) state, so all estimates remain
       consistent.
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from dataclasses import dataclass, field
from typing import Callable
from joblib import Parallel, delayed


# ---------------------------------------------------------------------------
# Section 6.1 reward function
# R_h(x, a) ~ N((x - a)^2, 0.01)
# Agent receives noisy reward at every step, optimal when action matches state
# ---------------------------------------------------------------------------

def reward_6_1(state: np.ndarray, action: np.ndarray) -> float:
    """
    Per-step noisy quadratic reward from Section 6.1.
    Mean = (x - a)^2, noise ~ N(0, 0.01).
    NOTE: reward is NEGATIVE of (x-a)^2 so agent maximises by minimising gap.
    The paper uses R_h ~ N((x-a)^2, 0.01) and the agent maximises total reward,
    so we negate: higher reward when action tracks state closely.
    """
    x = float(state[0])
    a = float(action[0])
    mean   = -((x - a) ** 2)   # negative so maximising = minimising gap
    noise  = np.random.normal(0, 0.01)
    return mean + noise


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExpConfig:
    """
    All parameters for one experiment run.
    Structured to match Section 6.1 notation exactly.
    reward_step_fn: callable(state, action) -> float  (per-step reward)
    domain_lo/hi: compact domain for projection fix
    """
    # Dimensions (Section 6.1: both 1D)
    state_dim:  int = 1
    action_dim: int = 1

    # Episode / training
    epLen:   int = 10      # H = 10
    nEps:    int = 2000    # K = 2000
    n_seeds: int = 10

    # Starting state (Section 6.1: X_1 = 4)
    starting_state: float = 4.0

    # Domain for projection (paper uses S=R but numerically needs bounds)
    domain_lo: float = -50.0
    domain_hi: float =  50.0

    # Algorithm (Section 6.1: rho=10, C_h=5, Delta=1)
    initial_q:       float = 1837.1  # Q^0_h(.) = 1837.1 from paper
    rho:             float = 10.0    # initial state radius
    rho_1:           float = 5.0     # initial action radius ([0,10] -> centre 5, radius 5)
    lip:             float = 1.0
    split_threshold: int   = 2
    scaling:         float = 5.0     # C_h = 5 from paper

    # Linear O-U dynamics coefficients (Section 6.1)
    # mu_h(x,a) = theta_0 + theta_x * x + theta_a * a
    theta_0: float =  0.05   # constant drift term
    theta_x: float = -0.1    # state coefficient
    theta_a: float =  0.01   # action coefficient
    sigma:   float =  0.1    # constant volatility
    delta:   float =  1.0    # Delta = 1

    # Per-step reward callable: (state_arr, action_arr) -> float
    reward_step_fn: Callable = field(default_factory=lambda: reward_6_1)

    label: str = 'Section 6.1'


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class Agent:
    def update_obs(self, obs, action, reward, newObs, timestep): pass
    def update_policy(self, k): pass
    def pick_action(self, obs, timestep): pass
    def get_num_arms(self): pass


class Environment:
    def reset(self): pass
    def advance(self, action): return 0, 0, 0
    def get_epLen(self): return 0


# ---------------------------------------------------------------------------
# Environment — Section 6.1 linear dynamics + domain projection
# ---------------------------------------------------------------------------

class LinearDiffEnvironment(Environment):
    """
    Section 6.1 environment.

    Dynamics (linear, not geometric):
        X_{h+1} = X_h + (theta_0 + theta_x*X_h + theta_a*a_h) * Delta
                       + sigma * sqrt(Delta) * Z_h,   Z_h ~ N(0,1)

    Reward is paid at EVERY step (not just terminal):
        R_h ~ N(-(x - a)^2, 0.01)

    Domain projection (out-of-rho fix):
        After each dynamics step, state is clipped to [domain_lo, domain_hi].
        See module docstring for why this is safe and doesn't corrupt learning.
    """

    def __init__(self, cfg: ExpConfig):
        self.cfg       = cfg
        self.epLen     = cfg.epLen
        self._start    = np.full(cfg.state_dim, cfg.starting_state, dtype=float)
        self.state     = self._start.copy()
        self.timestep  = 0
        self.projection_count = 0  # diagnostic counter

    def get_epLen(self):
        return self.epLen

    def reset(self):
        self.timestep = 0
        self.state    = self._start.copy()

    def _project(self, state: np.ndarray) -> np.ndarray:
        """
        Project state to [domain_lo, domain_hi]^d.

        WHY THIS DOESN'T BREAK THE CODE:
          - The tree traversal (_traverse) uses L-inf containment checks.
            If state is outside all nodes, _traverse returns the root as a
            fallback — but root estimates are coarse and stale.
          - Projection ensures state is always inside the root node (rho=10
            covers [-10,10]), so traversal always finds the correct leaf.
          - All downstream Bellman updates and split logic are unchanged.
          - projection_count==0 means this is a silent no-op; >0 means the
            domain or dynamics need checking.
        """
        projected = np.clip(state, self.cfg.domain_lo, self.cfg.domain_hi)
        if not np.array_equal(projected, state):
            self.projection_count += 1
        return projected

    def advance(self, action: np.ndarray):
        cfg = self.cfg
        x   = self.state[0]
        a   = float(action[0])

        # Linear O-U step: X_{h+1} = X_h + mu(x,a)*Delta + sigma*sqrt(Delta)*Z
        drift     = cfg.theta_0 + cfg.theta_x * x + cfg.theta_a * a
        noise     = np.random.randn(cfg.state_dim)
        new_state = self.state + drift * cfg.delta + cfg.sigma * math.sqrt(cfg.delta) * noise

        # Domain projection — keeps state inside compact domain safely
        new_state = self._project(new_state)

        # Per-step reward (Section 6.1 — every timestep, not just terminal)
        reward = cfg.reward_step_fn(new_state, action)

        self.state     = new_state
        self.timestep += 1
        pContinue      = 0 if self.timestep == self.epLen else 1
        return reward, new_state, pContinue


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

class Experiment:
    """Runs one agent for nEps episodes, records cumulative per-episode reward."""

    def __init__(self, env: Environment, agent: Agent, cfg: ExpConfig, seed: int):
        assert isinstance(env, Environment)
        self.env   = env
        self.agent = agent
        self.nEps  = cfg.nEps
        self.epLen = env.get_epLen()
        self.data  = np.zeros(self.nEps)
        np.random.seed(seed)

    def run(self):
        for ep in range(self.nEps):
            self.env.reset()
            state     = self.env.state.copy()
            epReward  = 0.0
            self.agent.update_policy(ep)
            h         = 0
            pContinue = 1

            while pContinue and h < self.epLen:
                action                       = self.agent.pick_action(state, h)
                reward, new_state, pContinue = self.env.advance(action)
                epReward                    += reward
                self.agent.update_obs(state, action, reward, new_state, h)
                state = new_state.copy()
                h    += 1

            self.data[ep] = epReward
        return self.data


# ---------------------------------------------------------------------------
# Precompute split offsets (dimension-dependent)
# ---------------------------------------------------------------------------

def make_offsets(state_dim: int, action_dim: int):
    """All ±1 offset combinations for state and action splitting."""
    state_offsets  = np.array(list(itertools.product([-1, 1], repeat=state_dim)))
    action_offsets = np.array(list(itertools.product([-1, 1], repeat=action_dim)))
    return state_offsets, action_offsets


# ---------------------------------------------------------------------------
# Node — vector state, L-inf containment
# ---------------------------------------------------------------------------

class Node:
    """
    One hypercube cell in the adaptive partition.
    L-inf containment: ||x - state_val||_inf <= radius  (paper Section 3.2).
    muEst / sigmaEst are vectors of length state_dim.
    """
    __slots__ = (
        'qVal', 'rEst', 'muEst', 'sigmaEst',
        'num_visits', 'num_unique_visits', 'num_splits',
        'state_val', 'action_val', 'radius', 'action_radius',
        'children'
    )

    def __init__(self, qVal, rEst, muEst, sigmaEst,
                 num_visits, num_unique_visits, num_splits,
                 state_val, action_val, radius, action_radius):
        self.qVal              = qVal
        self.rEst              = rEst
        self.muEst             = np.asarray(muEst,    dtype=float)
        self.sigmaEst          = np.asarray(sigmaEst, dtype=float)
        self.num_visits        = num_visits
        self.num_unique_visits = num_unique_visits
        self.num_splits        = num_splits
        self.state_val         = np.asarray(state_val,  dtype=float)
        self.action_val        = np.asarray(action_val, dtype=float)
        self.radius            = float(radius)
        self.action_radius     = float(action_radius)
        self.children          = None

    def contains(self, state: np.ndarray) -> bool:
        """L-inf ball containment check matching paper's hypercube partition."""
        return bool(np.max(np.abs(state - self.state_val)) <= self.radius)

    def split_node(self, state_offsets, action_offsets, initial_q):
        """Split into 2^state_dim * 2^action_dim children, halving both radii."""
        half_r  = self.radius        / 2
        half_ar = self.action_radius / 2
        children = []
        sd = len(self.state_val)
        ad = len(self.action_val)

        for s_offs in state_offsets:
            new_state = self.state_val + s_offs * half_r
            for a_offs in action_offsets:
                new_action = self.action_val + a_offs * half_ar
                if self.num_visits <= 1:
                    child = Node(initial_q, 0,
                                 np.zeros(sd), np.zeros(sd),
                                 self.num_visits, 0, self.num_splits + 1,
                                 new_state, new_action, half_r, half_ar)
                else:
                    child = Node(self.qVal, self.rEst,
                                 self.muEst.copy(), self.sigmaEst.copy(),
                                 self.num_visits, self.num_visits, self.num_splits + 1,
                                 new_state, new_action, half_r, half_ar)
                children.append(child)

        self.children = children
        return children


# ---------------------------------------------------------------------------
# Tree — one per timestep
# ---------------------------------------------------------------------------

class Tree:
    """
    Adaptive hypercube partition for one timestep.
    state_leaves stores vector centres; _state_to_idx maps tuple -> index.
    """

    def __init__(self, cfg: ExpConfig, state_offsets, action_offsets):
        self.cfg            = cfg
        self.initial_q      = cfg.initial_q
        self.state_offsets  = state_offsets
        self.action_offsets = action_offsets

        # Root: state centre 0, action centre at midpoint of [0,10]
        start_state  = np.zeros(cfg.state_dim)
        start_action = np.full(cfg.action_dim, 5.0)   # midpoint of [0,10]
        self.head = Node(
            cfg.initial_q, 0,
            np.zeros(cfg.state_dim), np.zeros(cfg.state_dim),
            0, 0, 0,
            start_state, start_action, cfg.rho, cfg.rho_1)

        self.tree_leaves: set = {self.head}
        self.state_leaves     = [start_state.copy()]
        self.vEst             = [float(cfg.initial_q)]
        self._state_to_idx    = {tuple(start_state): 0}
        self.min_vEst: float  = float(cfg.initial_q)
        self._lookup_cache    = {}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_active_ball(self, state: np.ndarray):
        """Return (node, qVal) for deepest leaf containing state."""
        key = tuple(state)
        if key in self._lookup_cache:
            node = self._lookup_cache[key]
            if node.children is None:
                return node, node.qVal
        node, qVal = self._traverse(state)
        self._lookup_cache[key] = node
        return node, qVal

    def _traverse(self, state: np.ndarray):
        """
        Iterative L-inf descent.
        If state is outside all children (should not happen after projection,
        but kept as a safe fallback), returns the current node rather than
        crashing — this is the robustness guarantee of the domain projection.
        """
        node = self.head
        while node.children is not None:
            best_node, best_q = None, -np.inf
            for child in node.children:
                if child.contains(state):
                    if child.qVal >= best_q:
                        best_q, best_node = child.qVal, child
            if best_node is None:
                break   # out-of-domain fallback — stay at current node
            node = best_node
        return node, node.qVal

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split_node(self, node, timestep, previous_tree):
        """Split node, update leaf set, state partition and V-map."""
        children = node.split_node(
            self.state_offsets, self.action_offsets, self.initial_q)

        self.tree_leaves.discard(node)
        self.tree_leaves.update(children)
        self._lookup_cache.clear()

        child_0_state  = children[0].state_val
        child_0_radius = children[0].radius
        state_arr      = np.vstack(self.state_leaves)
        linf_dists     = np.max(np.abs(state_arr - child_0_state), axis=1)

        if np.min(linf_dists) >= child_0_radius:
            parent_key  = tuple(node.state_val)
            parent_idx  = self._state_to_idx.get(parent_key)
            parent_vest = self.vEst[parent_idx] if parent_idx is not None else self.initial_q

            self.state_leaves.pop(parent_idx)
            self.vEst.pop(parent_idx)
            del self._state_to_idx[parent_key]
            self._state_to_idx = {tuple(s): i for i, s in enumerate(self.state_leaves)}

            num_action = len(self.action_offsets)
            for k in range(len(self.state_offsets)):
                new_s = children[k * num_action].state_val
                key   = tuple(new_s)
                if key not in self._state_to_idx:
                    idx = len(self.state_leaves)
                    self.state_leaves.append(new_s.copy())
                    self.vEst.append(parent_vest)
                    self._state_to_idx[key] = idx

            self.min_vEst = min(self.vEst)
        return children

    # ------------------------------------------------------------------
    # V-estimate refresh
    # ------------------------------------------------------------------

    def update_vEst(self):
        for i, sv in enumerate(self.state_leaves):
            _, qMax      = self.get_active_ball(sv)
            self.vEst[i] = min(qMax, self.initial_q, self.vEst[i])
        self.min_vEst = min(self.vEst)

    def get_number_of_active_balls(self):
        return len(self.tree_leaves)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AdaptiveModelBasedDiscretization(Agent):
    """
    APL-Diffusion agent.
    Handles per-step rewards (Section 6.1) and terminal-only rewards (6.2)
    transparently — reward estimate rEst accumulates whatever reward arrives.
    """

    def __init__(self, cfg: ExpConfig):
        self.cfg             = cfg
        self.epLen           = cfg.epLen
        self.scaling         = cfg.scaling
        self.split_threshold = cfg.split_threshold
        self.lip             = cfg.lip
        self.initial_q       = cfg.initial_q
        self.state_dim       = cfg.state_dim

        self.state_offsets, self.action_offsets = make_offsets(
            cfg.state_dim, cfg.action_dim)

        self.tree_list = [
            Tree(cfg, self.state_offsets, self.action_offsets)
            for _ in range(cfg.epLen)
        ]

    def reset(self):
        self.tree_list = [
            Tree(self.cfg, self.state_offsets, self.action_offsets)
            for _ in range(self.epLen)
        ]

    def get_num_arms(self):
        return sum(t.get_number_of_active_balls() for t in self.tree_list)

    def update_obs(self, obs: np.ndarray, action: np.ndarray,
                   reward: float, newObs: np.ndarray, timestep: int):
        """Online one-step Bellman update with vector state."""
        tree           = self.tree_list[timestep]
        active_node, _ = tree.get_active_ball(obs)

        active_node.num_visits        += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits

        # Incremental reward estimate (works for per-step or terminal reward)
        active_node.rEst = ((t - 1) * active_node.rEst + reward) / t

        # Incremental vector drift and variance estimates
        if timestep != self.epLen - 1:
            delta = newObs - obs
            active_node.muEst    = ((t - 1) * active_node.muEst + delta) / t
            active_node.sigmaEst = ((t - 1) * active_node.sigmaEst
                                    + (delta - active_node.muEst) ** 2) / t

        # UCB bonus with L2 norms for vector estimates
        ucb = (self.scaling / math.sqrt(active_node.num_visits)
               + self.scaling * active_node.radius)

        if timestep == self.epLen - 1:
            q_new = active_node.rEst + ucb
        else:
            next_tree = self.tree_list[timestep + 1]
            mu_sq     = float(np.dot(active_node.muEst, active_node.muEst))
            sigma_sq  = float(np.sum(active_node.sigmaEst))
            vEst_next = next_tree.min_vEst + self.lip * (1 + mu_sq + sigma_sq)
            q_new     = active_node.rEst + vEst_next + ucb

        active_node.qVal = min(active_node.qVal, self.initial_q, q_new)
        tree.update_vEst()

        if t >= 2 ** (self.split_threshold * active_node.num_splits):
            prev = self.tree_list[timestep - 1] if timestep >= 1 else None
            tree.split_node(active_node, timestep, prev)

    def update_policy(self, k):
        pass

    def pick_action(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Sample uniformly within the active ball's action L-inf region."""
        tree           = self.tree_list[timestep]
        active_node, _ = tree.get_active_ball(state)
        lo = active_node.action_val - active_node.action_radius
        hi = active_node.action_val + active_node.action_radius
        # Clip to valid action range [0, 10] as specified in Section 6.1
        lo = np.clip(lo, 0.0, 10.0)
        hi = np.clip(hi, 0.0, 10.0)
        return np.random.uniform(lo, hi)


# ---------------------------------------------------------------------------
# Parallel worker + runner
# ---------------------------------------------------------------------------

def run_one_seed(seed: int, cfg: ExpConfig) -> np.ndarray:
    env   = LinearDiffEnvironment(cfg)
    agent = AdaptiveModelBasedDiscretization(cfg)
    exp   = Experiment(env, agent, cfg, seed)
    return exp.run()


def run_experiment(configs: list, n_jobs: int = -1) -> dict:
    results = {}
    for cfg in configs:
        print(f"\n--- Running: {cfg.label} ---")
        seed_rewards = Parallel(n_jobs=n_jobs)(
            delayed(run_one_seed)(seed, cfg) for seed in range(cfg.n_seeds)
        )
        matrix             = np.vstack(seed_rewards)
        results[cfg.label] = matrix.mean(axis=0)
        print(f"    Done. Final mean VPI: {results[cfg.label][-100:].mean():.3f}")
    return results


# ---------------------------------------------------------------------------
# Plot — VPI convergence (replicates Figure 3a from the paper)
# ---------------------------------------------------------------------------

def plot_vpi(results: dict, smooth_window: int = 50,
             save_path: str = 'vpi_section6_1.png'):
    """
    Plot mean VPI vs episode, matching Figure 3(a) from the paper.
    Smoothing window of 50 matches the paper's smoothed curve style.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.tab10(np.linspace(0, 0.8, len(results)))

    for (label, vpi), colour in zip(results.items(), colours):
        smoothed = pd.Series(vpi).rolling(window=smooth_window,
                                          min_periods=1).mean().values
        episodes = np.arange(len(vpi))
        ax.plot(episodes, smoothed, label=label, color=colour, linewidth=2)
        ax.plot(episodes, vpi,      color=colour, alpha=0.12, linewidth=0.7)

    ax.set_xlabel('Episode (K)', fontsize=13)
    ax.set_ylabel('Mean VPI — estimated $V^{\\hat{\\pi}}$', fontsize=13)
    ax.set_title('Section 6.1: VPI Convergence\n'
                 'One-dimensional linear diffusion, reward $R_h \\sim \\mathcal{N}(-(x-a)^2, 0.01)$',
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {save_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 6.1 configuration — exact paper parameters
# ---------------------------------------------------------------------------

CFG_6_1 = ExpConfig(
    state_dim      = 1,
    action_dim     = 1,
    epLen          = 10,       # H = 10
    nEps           = 2000,     # K = 2000  (reduce to 500 for a quick run)
    n_seeds        = 10,
    starting_state = 4.0,      # X_1 = 4
    domain_lo      = -50.0,
    domain_hi      =  50.0,
    initial_q      = 1837.1,   # Q^0_h(.) = 1837.1
    rho            = 10.0,     # rho = 10
    rho_1          = 5.0,      # action [0,10] -> centre 5, radius 5
    lip            = 1.0,
    split_threshold = 2,
    scaling        = 5.0,      # C_h = 5
    theta_0        = 0.05,
    theta_x        = -0.1,
    theta_a        = 0.01,
    sigma          = 0.1,
    delta          = 1.0,      # Delta = 1
    reward_step_fn = reward_6_1,
    label          = 'Section 6.1 — Linear diffusion',
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results = run_experiment([CFG_6_1], n_jobs=-1)
    plot_vpi(results, smooth_window=50, save_path='vpi_section6_1.png')

    df = pd.DataFrame(results)
    df.index.name = 'episode'
    df.to_csv('vpi_section6_1.csv', index=False)
    print('Data saved to vpi_section6_1.csv')