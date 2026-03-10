import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple
from joblib import Parallel, delayed
import time


def reward_6_1(state: np.ndarray, action: np.ndarray) -> float:
    """Compute noisy quadratic reward: R ~ N(-(x-a)^2, 0.01)."""
    x, a = state[0], action[0]
    diff = x - a
    return -(diff * diff) + np.random.normal(0.0, 0.01)


@dataclass
class ExpConfig:
    """Experiment configuration storing all hyperparameters."""
    state_dim: int = 1
    action_dim: int = 1
    epLen: int = 10
    nEps: int = 2000
    n_seeds: int = 10
    starting_state: float = 4.0
    domain_lo: float = -50.0
    domain_hi: float = 50.0
    initial_q: float = 1837.1
    rho: float = 10.0
    rho_1: float = 5.0
    lip: float = 1.0
    split_threshold: int = 2
    scaling: float = 5.0
    alpha: float = 0.5
    theta_0: float = 0.05
    theta_x: float = -0.1
    theta_a: float = 0.01
    sigma: float = 0.1
    delta: float = 1.0
    reward_step_fn: Callable = field(default_factory=lambda: reward_6_1)
    label: str = 'Section 6.1'
    use_bellman: bool = True  # True = Full Bellman, False = One-Step
    inherit_flag: bool = False  # Whether to inherit estimates on split
    _sigma_sqrt_delta: float = field(init=False, repr=False)

    def __post_init__(self):
        self._sigma_sqrt_delta = self.sigma * math.sqrt(self.delta)


class Agent:
    """Abstract base class for reinforcement learning agents."""
    __slots__ = ()

    def update_obs(self, obs, action, reward, newObs, timestep):
        pass

    def update_policy(self, k):
        pass

    def pick_action(self, obs, timestep):
        pass

    def get_num_arms(self):
        pass


class Environment:
    """Abstract base class for RL environments."""
    __slots__ = ()

    def reset(self):
        pass

    def advance(self, action):
        return 0, 0, 0

    def get_epLen(self):
        return 0


class LinearDiffEnvironment(Environment):
    """
    Section 6.1 environment with linear diffusion dynamics.

    Dynamics: X_{h+1} = X_h + (theta_0 + theta_x*X_h + theta_a*a_h)*Delta
                            + sigma*sqrt(Delta)*Z_h
    States are projected onto [domain_lo, domain_hi] to maintain compactness.
    """
    __slots__ = ('cfg', 'epLen', '_start', 'state', 'timestep',
                 'projection_count', '_sigma_sqrt_delta', '_domain_lo', '_domain_hi')

    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.epLen = cfg.epLen
        self._start = np.array([cfg.starting_state], dtype=np.float64)
        self.state = self._start.copy()
        self.timestep = 0
        self.projection_count = 0
        self._sigma_sqrt_delta = cfg._sigma_sqrt_delta
        self._domain_lo = cfg.domain_lo
        self._domain_hi = cfg.domain_hi

    def get_epLen(self) -> int:
        return self.epLen

    def reset(self) -> None:
        self.timestep = 0
        self.state = self._start.copy()

    def advance(self, action: np.ndarray) -> Tuple[float, np.ndarray, int]:
        """Execute one environment step, returning (reward, new_state, continue_flag)."""
        cfg = self.cfg
        x, a = self.state[0], action[0]

        drift = cfg.theta_0 + cfg.theta_x * x + cfg.theta_a * a
        noise = np.random.randn()
        new_x = x + drift * cfg.delta + self._sigma_sqrt_delta * noise

        if new_x < self._domain_lo:
            new_x = self._domain_lo
            self.projection_count += 1
        elif new_x > self._domain_hi:
            new_x = self._domain_hi
            self.projection_count += 1

        self.state[0] = new_x

        reward = cfg.reward_step_fn(np.array([x], dtype=np.float64), action)

        self.timestep += 1
        pContinue = 1 if self.timestep < self.epLen else 0

        return reward, self.state, pContinue


class Experiment:
    """Runs episodes and collects cumulative rewards for one agent-environment pair."""
    __slots__ = ('env', 'agent', 'nEps', 'epLen', 'data', 'arms', '_seed')

    def __init__(self, env: Environment, agent: Agent, cfg: ExpConfig, seed: int):
        self.env = env
        self.agent = agent
        self.nEps = cfg.nEps
        self.epLen = env.get_epLen()
        self.data = np.zeros(self.nEps, dtype=np.float64)
        self.arms = np.zeros(self.nEps, dtype=np.float64)
        self._seed = seed

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """Execute all episodes and return arrays of per-episode rewards and active balls."""
        np.random.seed(self._seed)

        env_reset = self.env.reset
        env_advance = self.env.advance
        agent_update_policy = self.agent.update_policy
        agent_pick_action = self.agent.pick_action
        agent_update_obs = self.agent.update_obs
        agent_get_num_arms = self.agent.get_num_arms
        env_state = self.env
        epLen = self.epLen
        data = self.data
        arms = self.arms

        for ep in range(self.nEps):
            env_reset()
            state = env_state.state.copy()
            epReward = 0.0
            agent_update_policy(ep)

            for h in range(epLen):
                action = agent_pick_action(state, h)
                reward, new_state, pContinue = env_advance(action)
                epReward += reward
                agent_update_obs(state, action, reward, new_state, h)

                if not pContinue:
                    break

                state[0] = new_state[0]

            data[ep] = epReward
            arms[ep] = agent_get_num_arms()

        return data, arms


class Node:
    """
    Represents a hypercube cell in the adaptive partition.

    Each node covers an L-inf ball in state-action space and maintains
    estimates for Q-value, reward, drift, and variance.
    """
    __slots__ = (
        'qVal', 'rEst', 'muEst', 'sigmaEst',
        'num_visits', 'num_unique_visits', 'num_splits',
        'state_val', 'action_val', 'radius', 'action_radius',
        'children', '_state_center', '_action_center'
    )

    def __init__(self, qVal: float, rEst: float,
                 muEst: np.ndarray, sigmaEst: np.ndarray,
                 num_visits: int, num_unique_visits: int, num_splits: int,
                 state_val: np.ndarray, action_val: np.ndarray,
                 radius: float, action_radius: float):
        self.qVal = qVal
        self.rEst = rEst
        self.muEst = muEst
        self.sigmaEst = sigmaEst
        self.num_visits = num_visits
        self.num_unique_visits = num_unique_visits
        self.num_splits = num_splits
        self.state_val = state_val
        self.action_val = action_val
        self.radius = radius
        self.action_radius = action_radius
        self.children: Optional[List['Node']] = None
        self._state_center = float(state_val[0])
        self._action_center = float(action_val[0])

    def contains_1d(self, state_scalar: float) -> bool:
        """Check if state lies within this node's L-inf ball (1D version)."""
        return abs(state_scalar - self._state_center) <= self.radius

    def contains(self, state: np.ndarray) -> bool:
        """Check if state lies within this node's L-inf ball."""
        return abs(state[0] - self._state_center) <= self.radius

    def split_node_1d(self, initial_q: float, inherit_flag: bool = False) -> List['Node']:
        """Split node into 4 children by halving state and action radii."""
        half_r = self.radius * 0.5
        half_ar = self.action_radius * 0.5
        sc = self._state_center
        ac = self._action_center

        children = []
        inherit = inherit_flag and self.num_visits > 1

        if inherit:
            qVal_init = self.qVal
            rEst_init = self.rEst
            muEst_init = self.muEst.copy()
            sigmaEst_init = self.sigmaEst.copy()
            visits_init = self.num_visits
            unique_init = self.num_visits
        else:
            qVal_init = initial_q
            rEst_init = 0.0
            muEst_init = np.zeros(1, dtype=np.float64)
            sigmaEst_init = np.zeros(1, dtype=np.float64)
            visits_init = self.num_visits
            unique_init = 0

        num_splits_new = self.num_splits + 1

        for s_sign in (-1, 1):
            new_sc = sc + s_sign * half_r
            new_state = np.array([new_sc], dtype=np.float64)

            for a_sign in (-1, 1):
                new_ac = ac + a_sign * half_ar
                new_action = np.array([new_ac], dtype=np.float64)

                child = Node(
                    qVal_init, rEst_init,
                    muEst_init if not inherit else muEst_init.copy(),
                    sigmaEst_init if not inherit else sigmaEst_init.copy(),
                    visits_init, unique_init, num_splits_new,
                    new_state, new_action, half_r, half_ar
                )
                children.append(child)

        self.children = children
        return children


class Tree:
    """
    Adaptive partition tree for one timestep.

    Maintains a hierarchical partition of state-action space where each leaf
    node tracks Q-value estimates. Supports dynamic refinement via splitting.
    """
    __slots__ = ('cfg', 'initial_q', 'head', 'tree_leaves',
                 'state_leaves', 'vEst', '_state_to_idx', 'min_vEst',
                 '_lookup_cache', '_vEst_dirty', 'inherit_flag')

    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.initial_q = cfg.initial_q
        self.inherit_flag = cfg.inherit_flag

        start_state = np.array([0.0], dtype=np.float64)
        start_action = np.array([5.0], dtype=np.float64)

        self.head = Node(
            cfg.initial_q, 0.0,
            np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64),
            0, 0, 0,
            start_state, start_action, cfg.rho, cfg.rho_1
        )

        self.tree_leaves: Dict[int, Node] = {id(self.head): self.head}
        self.state_leaves: List[float] = [0.0]
        self.vEst: List[float] = [float(cfg.initial_q)]
        self._state_to_idx: Dict[float, int] = {0.0: 0}
        self.min_vEst: float = float(cfg.initial_q)
        self._lookup_cache: Dict[float, Node] = {}
        self._vEst_dirty = False

    def get_active_ball_1d(self, state_scalar: float) -> Tuple[Node, float]:
        """Find the deepest leaf node containing the given state (1D version)."""
        cached = self._lookup_cache.get(state_scalar)
        if cached is not None and cached.children is None:
            return cached, cached.qVal

        node = self.head
        while node.children is not None:
            best_node = None
            best_q = -math.inf

            for child in node.children:
                if child.contains_1d(state_scalar):
                    if child.qVal >= best_q:
                        best_q = child.qVal
                        best_node = child

            if best_node is None:
                break
            node = best_node

        self._lookup_cache[state_scalar] = node
        return node, node.qVal

    def get_active_ball(self, state: np.ndarray) -> Tuple[Node, float]:
        """Find the deepest leaf node containing the given state."""
        return self.get_active_ball_1d(state[0])

    def split_node_1d(self, node: Node) -> List[Node]:
        """Split a node and update internal bookkeeping structures."""
        children = node.split_node_1d(self.initial_q, self.inherit_flag)

        del self.tree_leaves[id(node)]
        for child in children:
            self.tree_leaves[id(child)] = child

        self._lookup_cache.clear()

        child_0_center = children[0]._state_center
        child_0_radius = children[0].radius

        needs_new_states = True
        for sc in self.state_leaves:
            if abs(sc - child_0_center) < child_0_radius:
                needs_new_states = False
                break

        if needs_new_states:
            parent_center = node._state_center
            parent_idx = self._state_to_idx.get(parent_center)

            if parent_idx is not None:
                parent_vest = self.vEst[parent_idx]

                self.state_leaves.pop(parent_idx)
                self.vEst.pop(parent_idx)
                del self._state_to_idx[parent_center]

                self._state_to_idx = {s: i for i, s in enumerate(self.state_leaves)}

                seen_states = set()
                for child in children:
                    sc = child._state_center
                    if sc not in seen_states and sc not in self._state_to_idx:
                        seen_states.add(sc)
                        idx = len(self.state_leaves)
                        self.state_leaves.append(sc)
                        self.vEst.append(parent_vest)
                        self._state_to_idx[sc] = idx

                self.min_vEst = min(self.vEst) if self.vEst else self.initial_q

        self._vEst_dirty = True
        return children

    def update_vEst(self) -> None:
        """Refresh value estimates for all tracked state centers."""
        if not self._vEst_dirty and len(self.tree_leaves) == 1:
            return

        initial_q = self.initial_q
        min_v = math.inf

        for i, sc in enumerate(self.state_leaves):
            _, qMax = self.get_active_ball_1d(sc)
            v = min(qMax, initial_q, self.vEst[i])
            self.vEst[i] = v
            if v < min_v:
                min_v = v

        self.min_vEst = min_v if min_v != math.inf else initial_q
        self._vEst_dirty = False

    def get_number_of_active_balls(self) -> int:
        """Return the number of leaf nodes in the partition."""
        return len(self.tree_leaves)


class AdaptiveModelBasedDiscretization(Agent):
    """
    APL-Diffusion agent implementing adaptive partition-based learning.

    Maintains a separate partition tree for each timestep and performs
    Bellman updates with UCB exploration bonuses. Nodes split when
    sufficiently visited to refine the state-action discretization.
    
    Parameters:
        use_bellman: If True, uses full Bellman updates with value propagation.
                     If False, uses one-step updates (reward only).
    """
    __slots__ = ('cfg', 'epLen', 'scaling', 'alpha', 'split_threshold', 'lip',
                 'initial_q', 'state_dim', 'tree_list', '_split_thresholds',
                 'use_bellman', 'inherit_flag')

    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.epLen = cfg.epLen
        self.scaling = cfg.scaling
        self.alpha = cfg.alpha
        self.split_threshold = cfg.split_threshold
        self.lip = cfg.lip
        self.initial_q = cfg.initial_q
        self.state_dim = cfg.state_dim
        self.use_bellman = cfg.use_bellman
        self.inherit_flag = cfg.inherit_flag

        self._split_thresholds = [
            2 ** (cfg.split_threshold * i) for i in range(21)
        ]

        self.tree_list = [Tree(cfg) for _ in range(cfg.epLen)]

    def reset(self) -> None:
        """Reset all partition trees to initial state."""
        self.tree_list = [Tree(self.cfg) for _ in range(self.epLen)]

    def get_num_arms(self) -> int:
        """Return total number of leaf nodes across all timesteps."""
        return sum(t.get_number_of_active_balls() for t in self.tree_list)

    def update_obs(self, obs: np.ndarray, action: np.ndarray,
                   reward: float, newObs: np.ndarray, timestep: int) -> None:
        """Update estimates based on observed transition and possibly split."""
        tree = self.tree_list[timestep]
        obs_scalar = obs[0]
        active_node, _ = tree.get_active_ball_1d(obs_scalar)

        active_node.num_visits += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits

        active_node.rEst = ((t - 1) * active_node.rEst + reward) / t

        is_terminal = timestep == self.epLen - 1
        if not is_terminal:
            delta = newObs[0] - obs_scalar
            old_mu = active_node.muEst[0]
            new_mu = ((t - 1) * old_mu + delta) / t
            active_node.muEst[0] = new_mu
            active_node.sigmaEst[0] = ((t - 1) * active_node.sigmaEst[0]
                                       + (delta - new_mu) ** 2) / t

        scaling = self.scaling
        ucb = scaling / (active_node.num_visits ** self.alpha) + scaling * active_node.radius

        if is_terminal:
            q_new = active_node.rEst + ucb
        else:
            if self.use_bellman:
                # Full Bellman update: use value function from next timestep
                next_tree = self.tree_list[timestep + 1]
                mu_sq = active_node.muEst[0] ** 2
                sigma_sq = active_node.sigmaEst[0]
                vEst_next = next_tree.min_vEst + self.lip * (1.0 + mu_sq + sigma_sq)
                q_new = active_node.rEst + vEst_next + ucb
            else:
                # One-step update: only use immediate reward (no value propagation)
                q_new = active_node.rEst + ucb

        active_node.qVal = min(active_node.qVal, self.initial_q, q_new)
        tree.update_vEst()

        num_splits = active_node.num_splits
        if num_splits < len(self._split_thresholds):
            threshold = self._split_thresholds[num_splits]
        else:
            threshold = 2 ** (self.split_threshold * num_splits)

        if t >= threshold:
            tree.split_node_1d(active_node)

    def update_policy(self, k: int) -> None:
        """Policy update hook (not used in this algorithm)."""
        pass

    def pick_action(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Select action uniformly from the active node's action region."""
        tree = self.tree_list[timestep]
        active_node, _ = tree.get_active_ball_1d(state[0])

        ac = active_node._action_center
        ar = active_node.action_radius

        lo = max(0.0, ac - ar)
        hi = min(10.0, ac + ar)

        return np.array([np.random.uniform(lo, hi)], dtype=np.float64)


def run_one_seed(seed: int, cfg: ExpConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Execute a single experiment run with the given random seed."""
    env = LinearDiffEnvironment(cfg)
    agent = AdaptiveModelBasedDiscretization(cfg)
    exp = Experiment(env, agent, cfg, seed)
    rewards, arms = exp.run()
    return rewards, arms


def run_one_seed_timed(seed: int, cfg: ExpConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """Execute a single experiment run with timing."""
    start_time = time.time()
    env = LinearDiffEnvironment(cfg)
    agent = AdaptiveModelBasedDiscretization(cfg)
    exp = Experiment(env, agent, cfg, seed)
    rewards, arms = exp.run()
    duration = time.time() - start_time
    return rewards, arms, duration


def run_experiment(configs: List[ExpConfig], n_jobs: int = -1) -> Dict[str, Dict[str, np.ndarray]]:
    """Run experiments for all configurations, parallelizing across seeds."""
    results = {}

    for cfg in configs:
        print(f"\n--- Running: {cfg.label} ---")

        seed_runs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_one_seed)(seed, cfg) for seed in range(cfg.n_seeds)
        )

        reward_matrix = np.vstack([r for r, a in seed_runs])
        arm_matrix = np.vstack([a for r, a in seed_runs])

        results[cfg.label] = {
            "vpi": reward_matrix.mean(axis=0),
            "vpi_std": reward_matrix.std(axis=0),
            "arms": arm_matrix.mean(axis=0)
        }

        final_mean = results[cfg.label]["vpi"][-100:].mean()
        print(f"    Done. Final mean VPI: {final_mean:.3f}")

    return results


def run_comparative_study(n_jobs: int = -1) -> Dict[str, Dict]:
    """
    Run comparative study between Full Bellman and One-Step update strategies.
    
    Returns a dictionary with results for both strategies including:
    - VPI (value per iteration)
    - Arms (number of active balls)
    - Timing information
    """
    results = {}
    
    # Configuration for Full Bellman (High precision, Higher cost)
    cfg_bellman = ExpConfig(
        state_dim=1,
        action_dim=1,
        epLen=10,
        nEps=2000,
        n_seeds=10,
        starting_state=4.0,
        domain_lo=-50.0,
        domain_hi=50.0,
        initial_q=1837.1,
        rho=10.0,
        rho_1=5.0,
        lip=1.0,
        split_threshold=2,
        scaling=5.0,
        alpha=0.5,
        theta_0=0.05,
        theta_x=-0.1,
        theta_a=0.01,
        sigma=0.1,
        delta=1.0,
        reward_step_fn=reward_6_1,
        label='Full Bellman Update',
        use_bellman=True,
        inherit_flag=False,
    )
    
    # Configuration for One-Step (Lower precision, Lower cost)
    cfg_one_step = ExpConfig(
        state_dim=1,
        action_dim=1,
        epLen=10,
        nEps=2000,
        n_seeds=10,
        starting_state=4.0,
        domain_lo=-50.0,
        domain_hi=50.0,
        initial_q=1837.1,
        rho=10.0,
        rho_1=5.0,
        lip=1.0,
        split_threshold=2,
        scaling=5.0,
        alpha=0.5,
        theta_0=0.05,
        theta_x=-0.1,
        theta_a=0.01,
        sigma=0.1,
        delta=1.0,
        reward_step_fn=reward_6_1,
        label='One-Step Update',
        use_bellman=False,
        inherit_flag=False,
    )
    
    configs = [cfg_bellman, cfg_one_step]
    
    for cfg in configs:
        print(f"\n--- Running Comparative Study: {cfg.label} ---")
        print(f"    use_bellman={cfg.use_bellman}, inherit_flag={cfg.inherit_flag}")
        
        start_time = time.time()
        
        # Run experiments with timing
        seed_runs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_one_seed_timed)(seed, cfg) for seed in range(cfg.n_seeds)
        )
        
        total_time = time.time() - start_time
        
        reward_matrix = np.vstack([r for r, a, t in seed_runs])
        arm_matrix = np.vstack([a for r, a, t in seed_runs])
        seed_times = np.array([t for r, a, t in seed_runs])
        
        results[cfg.label] = {
            "vpi": reward_matrix.mean(axis=0),
            "vpi_std": reward_matrix.std(axis=0),
            "arms": arm_matrix.mean(axis=0),
            "arms_std": arm_matrix.std(axis=0),
            "total_time": total_time,
            "mean_seed_time": seed_times.mean(),
            "std_seed_time": seed_times.std(),
            "use_bellman": cfg.use_bellman,
        }
        
        final_mean = results[cfg.label]["vpi"][-100:].mean()
        final_std = results[cfg.label]["vpi_std"][-100:].mean()
        print(f"    Done. Final mean VPI: {final_mean:.3f} ± {final_std:.3f}")
        print(f"    Total time: {total_time:.2f}s, Mean per seed: {seed_times.mean():.2f}s")
    
    return results


def plot_vpi(results: Dict[str, Dict[str, np.ndarray]], smooth_window: int = 50,
             save_path: str = 'vpi_section6_1.png') -> None:
    """Plot VPI convergence curves with smoothing."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.tab10(np.linspace(0, 0.8, len(results)))

    for (label, series), colour in zip(results.items(), colours):
        vpi = series["vpi"]
        episodes = np.arange(len(vpi))

        cumsum = np.cumsum(np.insert(vpi, 0, 0))
        smoothed = np.empty_like(vpi)
        for i in range(len(vpi)):
            start = max(0, i - smooth_window + 1)
            smoothed[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)

        ax.plot(episodes, smoothed, label=label, color=colour, linewidth=2)
        ax.plot(episodes, vpi, color=colour, alpha=0.12, linewidth=0.7)

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


def plot_comparative_study(results: Dict[str, Dict], smooth_window: int = 50,
                           save_path: str = 'comparative_study.png') -> None:
    """Plot comparative study results: VPI and Arms side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colours = {'Full Bellman Update': 'tab:blue', 'One-Step Update': 'tab:orange'}
    
    # Plot 1: VPI Convergence
    ax1 = axes[0]
    for label, series in results.items():
        vpi = series["vpi"]
        episodes = np.arange(len(vpi))
        
        cumsum = np.cumsum(np.insert(vpi, 0, 0))
        smoothed = np.empty_like(vpi)
        for i in range(len(vpi)):
            start = max(0, i - smooth_window + 1)
            smoothed[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)
        
        color = colours.get(label, 'tab:gray')
        ax1.plot(episodes, smoothed, label=label, color=color, linewidth=2)
        ax1.plot(episodes, vpi, color=color, alpha=0.12, linewidth=0.7)
    
    ax1.set_xlabel('Episode (K)', fontsize=12)
    ax1.set_ylabel('Mean VPI', fontsize=12)
    ax1.set_title('VPI Convergence Comparison', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of Active Balls
    ax2 = axes[1]
    for label, series in results.items():
        arms = series["arms"]
        episodes = np.arange(len(arms))
        color = colours.get(label, 'tab:gray')
        ax2.plot(episodes, arms, label=label, color=color, linewidth=2)
    
    ax2.set_xlabel('Episode (K)', fontsize=12)
    ax2.set_ylabel('Number of Active Balls', fontsize=12)
    ax2.set_title('Partition Complexity', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Timing Comparison (Bar Chart)
    ax3 = axes[2]
    labels = list(results.keys())
    times = [results[l]["mean_seed_time"] for l in labels]
    time_stds = [results[l]["std_seed_time"] for l in labels]
    bar_colors = [colours.get(l, 'tab:gray') for l in labels]
    
    bars = ax3.bar(labels, times, yerr=time_stds, capsize=5, color=bar_colors, alpha=0.8)
    ax3.set_ylabel('Time per Seed (seconds)', fontsize=12)
    ax3.set_title('Computational Cost', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax3.annotate(f'{time_val:.2f}s',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)
    
    fig.suptitle('Bellman vs One-Step Update: Comparative Study\n'
                 'Linear Diffusion Environment (Section 6.1)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Comparative study plot saved to {save_path}')
    plt.close(fig)


def print_comparative_summary(results: Dict[str, Dict]) -> None:
    """Print a summary table of the comparative study results."""
    print("\n" + "=" * 70)
    print("COMPARATIVE STUDY SUMMARY: Full Bellman vs One-Step Update")
    print("=" * 70)
    
    headers = ["Metric", "Full Bellman", "One-Step", "Difference"]
    print(f"\n{headers[0]:<30} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15}")
    print("-" * 70)
    
    bellman = results.get('Full Bellman Update', {})
    one_step = results.get('One-Step Update', {})
    
    if bellman and one_step:
        # Final VPI (last 100 episodes)
        bellman_vpi = bellman["vpi"][-100:].mean()
        one_step_vpi = one_step["vpi"][-100:].mean()
        diff_vpi = bellman_vpi - one_step_vpi
        print(f"{'Final VPI (last 100 eps)':<30} {bellman_vpi:<15.3f} {one_step_vpi:<15.3f} {diff_vpi:+.3f}")
        
        # Final Arms
        bellman_arms = bellman["arms"][-1]
        one_step_arms = one_step["arms"][-1]
        diff_arms = bellman_arms - one_step_arms
        print(f"{'Final Active Balls':<30} {bellman_arms:<15.0f} {one_step_arms:<15.0f} {diff_arms:+.0f}")
        
        # Timing
        bellman_time = bellman["mean_seed_time"]
        one_step_time = one_step["mean_seed_time"]
        speedup = bellman_time / one_step_time if one_step_time > 0 else float('inf')
        print(f"{'Mean Time per Seed (s)':<30} {bellman_time:<15.2f} {one_step_time:<15.2f} {speedup:.2f}x")
        
        # Convergence speed (episode to reach 90% of final performance)
        def get_convergence_episode(vpi, threshold_pct=0.9):
            final_val = vpi[-100:].mean()
            threshold = final_val * threshold_pct
            for i, v in enumerate(vpi):
                if v >= threshold:
                    return i
            return len(vpi)
        
        bellman_conv = get_convergence_episode(bellman["vpi"])
        one_step_conv = get_convergence_episode(one_step["vpi"])
        diff_conv = bellman_conv - one_step_conv
        print(f"{'Episodes to 90% Final VPI':<30} {bellman_conv:<15d} {one_step_conv:<15d} {diff_conv:+d}")
    
    print("=" * 70)
    print("\nInterpretation:")
    print("- Full Bellman: Uses complete value function propagation (higher precision)")
    print("- One-Step: Uses only immediate rewards (lower computational cost)")
    print("=" * 70 + "\n")


CFG_6_1 = ExpConfig(
    state_dim=1,
    action_dim=1,
    epLen=10,
    nEps=2000,
    n_seeds=10,
    starting_state=4.0,
    domain_lo=-50.0,
    domain_hi=50.0,
    initial_q=1837.1,
    rho=10.0,
    rho_1=5.0,
    lip=1.0,
    split_threshold=2,
    scaling=5.0,
    alpha=0.5,
    theta_0=0.05,
    theta_x=-0.1,
    theta_a=0.01,
    sigma=0.1,
    delta=1.0,
    reward_step_fn=reward_6_1,
    label='Section 6.1 — Linear diffusion',
    use_bellman=True,
    inherit_flag=False,
)

# ============================================================================
# PART 3: Bellman Tuning Study
# ============================================================================

def run_bellman_tuning_study(n_jobs: int = -1) -> Dict[str, Dict]:
    """
    Test different Bellman hyperparameter configurations to see if tuning helps.
    """
    results = {}
    
    configs = [
        # Baseline One-Step (for comparison)
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=2000, n_seeds=10,
            starting_state=4.0, domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.05, theta_x=-0.1, theta_a=0.01, sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='One-Step (Baseline)',
            use_bellman=False,
            initial_q=1837.1,
            lip=1.0,
            scaling=5.0,
            inherit_flag=False,
        ),
        # Original Bellman (problematic)
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=2000, n_seeds=10,
            starting_state=4.0, domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.05, theta_x=-0.1, theta_a=0.01, sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='Bellman (Original)',
            use_bellman=True,
            initial_q=1837.1,
            lip=1.0,
            scaling=5.0,
            inherit_flag=False,
        ),
        # Tuned: Lower initial_q
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=2000, n_seeds=10,
            starting_state=4.0, domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.05, theta_x=-0.1, theta_a=0.01, sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='Bellman (init_q=10)',
            use_bellman=True,
            initial_q=10.0,
            lip=1.0,
            scaling=5.0,
            inherit_flag=False,
        ),
        # Tuned: Zero Lipschitz term
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=2000, n_seeds=10,
            starting_state=4.0, domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.05, theta_x=-0.1, theta_a=0.01, sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='Bellman (lip=0)',
            use_bellman=True,
            initial_q=1837.1,
            lip=0.0,
            scaling=5.0,
            inherit_flag=False,
        ),
        # Tuned: Both fixes
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=2000, n_seeds=10,
            starting_state=4.0, domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.05, theta_x=-0.1, theta_a=0.01, sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='Bellman (q=10, lip=0)',
            use_bellman=True,
            initial_q=10.0,
            lip=0.0,
            scaling=5.0,
            inherit_flag=False,
        ),
    ]
    
    for cfg in configs:
        print(f"\n--- Tuning Study: {cfg.label} ---")
        
        start_time = time.time()
        seed_runs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_one_seed)(seed, cfg) for seed in range(cfg.n_seeds)
        )
        total_time = time.time() - start_time
        
        reward_matrix = np.vstack([r for r, a in seed_runs])
        arm_matrix = np.vstack([a for r, a in seed_runs])
        
        results[cfg.label] = {
            "vpi": reward_matrix.mean(axis=0),
            "vpi_std": reward_matrix.std(axis=0),
            "arms": arm_matrix.mean(axis=0),
            "total_time": total_time,
        }
        
        final_mean = results[cfg.label]["vpi"][-100:].mean()
        final_std = results[cfg.label]["vpi_std"][-100:].mean()
        print(f"    Final VPI: {final_mean:.3f} ± {final_std:.3f}, Time: {total_time:.2f}s")
    
    return results


def plot_tuning_study(results: Dict[str, Dict], smooth_window: int = 50,
                      save_path: str = 'bellman_tuning.png') -> None:
    """Plot tuning study results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Panel 1: VPI Convergence
    ax1 = axes[0]
    for (label, series), color in zip(results.items(), colors):
        vpi = series["vpi"]
        episodes = np.arange(len(vpi))
        
        cumsum = np.cumsum(np.insert(vpi, 0, 0))
        smoothed = np.empty_like(vpi)
        for i in range(len(vpi)):
            start = max(0, i - smooth_window + 1)
            smoothed[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)
        
        ax1.plot(episodes, smoothed, label=label, color=color, linewidth=2)
    
    ax1.set_xlabel('Episode (K)', fontsize=12)
    ax1.set_ylabel('Mean VPI', fontsize=12)
    ax1.set_title('VPI Convergence: Bellman Hyperparameter Tuning', fontsize=12)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Final VPI Bar Chart
    ax2 = axes[1]
    labels = list(results.keys())
    final_vpis = [results[l]["vpi"][-100:].mean() for l in labels]
    final_stds = [results[l]["vpi_std"][-100:].mean() for l in labels]
    
    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, final_vpis, yerr=final_stds, capsize=4, 
                   color=colors, alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Final VPI (last 100 episodes)', fontsize=12)
    ax2.set_title('Final Performance Comparison', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Tuning study plot saved to {save_path}')
    plt.close(fig)


# ============================================================================
# PART 4: Delayed Reward Environment (Bellman Justification)
# ============================================================================

class DelayedRewardEnvironment(Environment):
    """
    Environment with delayed/sparse rewards that require Bellman updates.
    
    The agent must learn to take actions that don't yield immediate reward
    but position the state for high terminal reward.
    
    Reward types:
    - 'terminal': Only reward at final timestep (pure goal-reaching)
    - 'delayed_goal': Small action cost + terminal goal reward
    - 'original': Use the standard reward function
    """
    __slots__ = ('cfg', 'epLen', '_start', 'state', 'timestep',
                 'projection_count', '_sigma_sqrt_delta', '_domain_lo', 
                 '_domain_hi', 'goal', 'reward_type')

    def __init__(self, cfg: ExpConfig, goal: float = 0.0, reward_type: str = 'terminal'):
        self.cfg = cfg
        self.epLen = cfg.epLen
        self._start = np.array([cfg.starting_state], dtype=np.float64)
        self.state = self._start.copy()
        self.timestep = 0
        self.projection_count = 0
        self._sigma_sqrt_delta = cfg._sigma_sqrt_delta
        self._domain_lo = cfg.domain_lo
        self._domain_hi = cfg.domain_hi
        self.goal = goal
        self.reward_type = reward_type

    def get_epLen(self) -> int:
        return self.epLen

    def reset(self) -> None:
        self.timestep = 0
        self.state = self._start.copy()

    def advance(self, action: np.ndarray) -> Tuple[float, np.ndarray, int]:
        cfg = self.cfg
        x, a = self.state[0], action[0]

        # Same dynamics as original environment
        drift = cfg.theta_0 + cfg.theta_x * x + cfg.theta_a * a
        noise = np.random.randn()
        new_x = x + drift * cfg.delta + self._sigma_sqrt_delta * noise

        # Project to domain
        if new_x < self._domain_lo:
            new_x = self._domain_lo
            self.projection_count += 1
        elif new_x > self._domain_hi:
            new_x = self._domain_hi
            self.projection_count += 1

        self.state[0] = new_x

        # Compute reward based on type
        if self.reward_type == 'terminal':
            # Only reward at final timestep
            if self.timestep == self.epLen - 1:
                reward = -((new_x - self.goal) ** 2)
            else:
                reward = 0.0
                
        elif self.reward_type == 'delayed_goal':
            # Small action cost + terminal goal reward
            action_cost = -0.01 * (a ** 2)
            if self.timestep == self.epLen - 1:
                goal_reward = -((new_x - self.goal) ** 2)
                reward = goal_reward + action_cost
            else:
                reward = action_cost
                
        else:  # 'original'
            reward = cfg.reward_step_fn(np.array([x], dtype=np.float64), action)

        self.timestep += 1
        pContinue = 1 if self.timestep < self.epLen else 0

        return reward, self.state, pContinue


def run_delayed_reward_experiment(n_jobs: int = -1, goal: float = 0.0, 
                                   reward_type: str = 'terminal') -> Dict[str, Dict]:
    """
    Run experiment with delayed rewards where Bellman should outperform One-Step.
    
    Setup:
    - Start state: x=4.0
    - Goal state: x=0.0 (default)
    - Reward: Only at terminal step, based on distance to goal
    
    Why Bellman should win:
    - One-Step sees no reward signal until final step
    - Cannot learn which actions lead to good terminal states
    - Bellman propagates terminal value backward through time
    """
    results = {}
    
    configs = [
        # Bellman with tuned parameters
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=3000, n_seeds=10,
            starting_state=4.0, domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.05, theta_x=-0.1, theta_a=0.01, sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='Bellman (Delayed Reward)',
            use_bellman=True,
            initial_q=10.0,
            lip=0.1,
            scaling=5.0,
            inherit_flag=False,
        ),
        # One-Step for comparison
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=3000, n_seeds=10,
            starting_state=4.0, domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.05, theta_x=-0.1, theta_a=0.01, sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='One-Step (Delayed Reward)',
            use_bellman=False,
            initial_q=10.0,
            lip=0.0,
            scaling=5.0,
            inherit_flag=False,
        ),
    ]
    
    for cfg in configs:
        print(f"\n--- Delayed Reward Experiment: {cfg.label} ---")
        print(f"    Goal: {goal}, Start: {cfg.starting_state}, Reward type: {reward_type}")
        
        start_time = time.time()
        
        all_rewards = []
        all_arms = []
        
        for seed in range(cfg.n_seeds):
            np.random.seed(seed)
            
            # Use delayed reward environment
            env = DelayedRewardEnvironment(cfg, goal=goal, reward_type=reward_type)
            agent = AdaptiveModelBasedDiscretization(cfg)
            
            episode_rewards = []
            episode_arms = []
            
            for ep in range(cfg.nEps):
                env.reset()
                state = env.state.copy()
                ep_reward = 0.0
                agent.update_policy(ep)
                
                for h in range(cfg.epLen):
                    action = agent.pick_action(state, h)
                    reward, new_state, pContinue = env.advance(action)
                    ep_reward += reward
                    agent.update_obs(state, action, reward, new_state, h)
                    
                    if not pContinue:
                        break
                    state = new_state.copy()
                
                episode_rewards.append(ep_reward)
                episode_arms.append(agent.get_num_arms())
            
            all_rewards.append(episode_rewards)
            all_arms.append(episode_arms)
        
        total_time = time.time() - start_time
        
        reward_matrix = np.array(all_rewards)
        arm_matrix = np.array(all_arms)
        
        results[cfg.label] = {
            "vpi": reward_matrix.mean(axis=0),
            "vpi_std": reward_matrix.std(axis=0),
            "arms": arm_matrix.mean(axis=0),
            "total_time": total_time,
            "use_bellman": cfg.use_bellman,
        }
        
        final_mean = results[cfg.label]["vpi"][-100:].mean()
        final_std = results[cfg.label]["vpi_std"][-100:].mean()
        print(f"    Final VPI: {final_mean:.3f} ± {final_std:.3f}")
        print(f"    Time: {total_time:.2f}s")
    
    return results


def plot_delayed_reward_comparison(results: Dict[str, Dict], smooth_window: int = 50,
                                    save_path: str = 'delayed_reward_comparison.png') -> None:
    """Plot comparison for delayed reward experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {
        'Bellman (Delayed Reward)': 'tab:blue',
        'One-Step (Delayed Reward)': 'tab:orange'
    }
    
    # Panel 1: VPI Convergence
    ax1 = axes[0]
    for label, series in results.items():
        vpi = series["vpi"]
        episodes = np.arange(len(vpi))
        
        # Smoothing
        cumsum = np.cumsum(np.insert(vpi, 0, 0))
        smoothed = np.empty_like(vpi)
        for i in range(len(vpi)):
            start = max(0, i - smooth_window + 1)
            smoothed[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)
        
        color = colors.get(label, 'tab:gray')
        ax1.plot(episodes, smoothed, label=label, color=color, linewidth=2)
        ax1.plot(episodes, vpi, color=color, alpha=0.15, linewidth=0.5)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Mean VPI', fontsize=12)
    ax1.set_title('VPI Convergence\n(Delayed/Sparse Reward)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Active Balls
    ax2 = axes[1]
    for label, series in results.items():
        arms = series["arms"]
        episodes = np.arange(len(arms))
        color = colors.get(label, 'tab:gray')
        ax2.plot(episodes, arms, label=label, color=color, linewidth=2)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Number of Active Balls', fontsize=12)
    ax2.set_title('Partition Complexity', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Final VPI Bar Chart
    ax3 = axes[2]
    labels = list(results.keys())
    final_vpis = [results[l]["vpi"][-100:].mean() for l in labels]
    final_stds = [results[l]["vpi_std"][-100:].mean() for l in labels]
    bar_colors = [colors.get(l, 'tab:gray') for l in labels]
    
    x_pos = np.arange(len(labels))
    bars = ax3.bar(x_pos, final_vpis, yerr=final_stds, capsize=5, 
                   color=bar_colors, alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Bellman', 'One-Step'], fontsize=11)
    ax3.set_ylabel('Final VPI (last 100 eps)', fontsize=12)
    ax3.set_title('Final Performance', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value annotations
    for bar, val, std in zip(bars, final_vpis, final_stds):
        ax3.annotate(f'{val:.2f}', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)
    
    fig.suptitle('Delayed Reward Environment: Bellman vs One-Step\n'
                 'Terminal reward only — Goal reaching from x=4 to x=0', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Delayed reward plot saved to {save_path}')
    plt.close(fig)


def print_delayed_reward_summary(results: Dict[str, Dict]) -> None:
    """Print summary for delayed reward experiment."""
    print("\n" + "=" * 70)
    print("DELAYED REWARD EXPERIMENT SUMMARY")
    print("=" * 70)
    
    bellman = results.get('Bellman (Delayed Reward)', {})
    one_step = results.get('One-Step (Delayed Reward)', {})
    
    if bellman and one_step:
        bellman_vpi = bellman["vpi"][-100:].mean()
        one_step_vpi = one_step["vpi"][-100:].mean()
        
        print(f"\n{'Method':<30} {'Final VPI':<15} {'Time (s)':<10}")
        print("-" * 55)
        print(f"{'Bellman (Delayed Reward)':<30} {bellman_vpi:<15.3f} {bellman['total_time']:<10.2f}")
        print(f"{'One-Step (Delayed Reward)':<30} {one_step_vpi:<15.3f} {one_step['total_time']:<10.2f}")
        print("-" * 55)
        
        if bellman_vpi > one_step_vpi:
            winner = "Bellman"
            margin = bellman_vpi - one_step_vpi
        else:
            winner = "One-Step"
            margin = one_step_vpi - bellman_vpi
        
        print(f"\nWinner: {winner} (by {margin:.3f})")
        
        if winner == "Bellman":
            print("\n✓ BELLMAN JUSTIFIED: Value propagation is necessary for")
            print("  delayed reward environments where immediate rewards are sparse.")
        else:
            print("\n✗ Unexpected: One-Step still wins. Consider:")
            print("  - Increasing episode length (epLen)")
            print("  - Making reward even sparser")
            print("  - Checking Bellman hyperparameters")
    
    print("=" * 70)

def run_challenging_delayed_reward(n_jobs: int = -1) -> Dict[str, Dict]:
    """
    Environment where Bellman MUST outperform One-Step.
    
    Key changes:
    - Goal AWAY from natural equilibrium
    - Larger action effect
    - Dynamics push AWAY from goal
    """
    results = {}
    
    configs = [
        # Bellman
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=3000, n_seeds=10,
            starting_state=0.0,      # Start at origin
            domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.0,             # No constant drift
            theta_x=0.1,             # POSITIVE feedback (unstable, pushes away from 0)
            theta_a=0.5,             # LARGE action effect (actions matter!)
            sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='Bellman (Challenging)',
            use_bellman=True,
            initial_q=10.0,
            lip=0.1,
            scaling=5.0,
            inherit_flag=False,
        ),
        # One-Step
        ExpConfig(
            state_dim=1, action_dim=1, epLen=10, nEps=3000, n_seeds=10,
            starting_state=0.0,
            domain_lo=-50.0, domain_hi=50.0,
            rho=10.0, rho_1=5.0, split_threshold=2, alpha=0.5,
            theta_0=0.0,
            theta_x=0.1,             # Same unstable dynamics
            theta_a=0.5,             # Same large action effect
            sigma=0.1, delta=1.0,
            reward_step_fn=reward_6_1,
            label='One-Step (Challenging)',
            use_bellman=False,
            initial_q=10.0,
            lip=0.0,
            scaling=5.0,
            inherit_flag=False,
        ),
    ]
    
    goal = 5.0  # Goal is AWAY from start, requires active control
    
    for cfg in configs:
        print(f"\n--- Challenging Delayed Reward: {cfg.label} ---")
        print(f"    Goal: {goal}, Start: {cfg.starting_state}")
        print(f"    theta_x: {cfg.theta_x} (unstable), theta_a: {cfg.theta_a} (strong)")
        
        start_time = time.time()
        
        all_rewards = []
        all_arms = []
        
        for seed in range(cfg.n_seeds):
            np.random.seed(seed)
            
            env = DelayedRewardEnvironment(cfg, goal=goal, reward_type='terminal')
            agent = AdaptiveModelBasedDiscretization(cfg)
            
            episode_rewards = []
            episode_arms = []
            
            for ep in range(cfg.nEps):
                env.reset()
                state = env.state.copy()
                ep_reward = 0.0
                agent.update_policy(ep)
                
                for h in range(cfg.epLen):
                    action = agent.pick_action(state, h)
                    reward, new_state, pContinue = env.advance(action)
                    ep_reward += reward
                    agent.update_obs(state, action, reward, new_state, h)
                    
                    if not pContinue:
                        break
                    state = new_state.copy()
                
                episode_rewards.append(ep_reward)
                episode_arms.append(agent.get_num_arms())
            
            all_rewards.append(episode_rewards)
            all_arms.append(episode_arms)
        
        total_time = time.time() - start_time
        
        reward_matrix = np.array(all_rewards)
        arm_matrix = np.array(all_arms)
        
        results[cfg.label] = {
            "vpi": reward_matrix.mean(axis=0),
            "vpi_std": reward_matrix.std(axis=0),
            "arms": arm_matrix.mean(axis=0),
            "total_time": total_time,
            "use_bellman": cfg.use_bellman,
        }
        
        final_mean = results[cfg.label]["vpi"][-100:].mean()
        final_std = results[cfg.label]["vpi_std"][-100:].mean()
        print(f"    Final VPI: {final_mean:.3f} ± {final_std:.3f}")
        print(f"    Time: {total_time:.2f}s")
    
    return results

if __name__ == '__main__':
    # ========================================
    # Part 1: Original Section 6.1 Experiment
    # ========================================
    print("\n" + "=" * 70)
    print("PART 1: Original Section 6.1 Experiment")
    print("=" * 70)
    
    results = run_experiment([CFG_6_1], n_jobs=-1)
    plot_vpi(results, smooth_window=50, save_path='vpi_section6_1.png')

    df_vpi = pd.DataFrame({k: v["vpi"] for k, v in results.items()})
    df_vpi.to_csv('vpi_section6_1.csv', index=False)

    df_arms = pd.DataFrame({k: v["arms"] for k, v in results.items()})
    df_arms.to_csv('arms_section6_1.csv', index=False)
    print('Data saved to vpi_section6_1.csv and arms_section6_1.csv')

    # ========================================
    # Part 2: Comparative Study (Bellman vs One-Step)
    # ========================================
    print("\n" + "=" * 70)
    print("PART 2: Comparative Study - Full Bellman vs One-Step Update")
    print("=" * 70)
    
    comparative_results = run_comparative_study(n_jobs=-1)
    plot_comparative_study(comparative_results, smooth_window=50, 
                           save_path='comparative_study.png')
    print_comparative_summary(comparative_results)
    
    df_comp_vpi = pd.DataFrame({k: v["vpi"] for k, v in comparative_results.items()})
    df_comp_vpi.to_csv('comparative_vpi.csv', index=False)
    
    df_comp_arms = pd.DataFrame({k: v["arms"] for k, v in comparative_results.items()})
    df_comp_arms.to_csv('comparative_arms.csv', index=False)

    # ========================================
    # Part 3: Bellman Hyperparameter Tuning
    # ========================================
    print("\n" + "=" * 70)
    print("PART 3: Bellman Hyperparameter Tuning")
    print("=" * 70)
    
    tuning_results = run_bellman_tuning_study(n_jobs=-1)
    plot_tuning_study(tuning_results, smooth_window=50, save_path='bellman_tuning.png')
    
    df_tuning = pd.DataFrame({k: v["vpi"] for k, v in tuning_results.items()})
    df_tuning.to_csv('bellman_tuning_vpi.csv', index=False)

    # ========================================
    # Part 4: Delayed Reward Experiment (Easy)
    # ========================================
    print("\n" + "=" * 70)
    print("PART 4: Delayed Reward Experiment (Easy - Natural Dynamics)")
    print("=" * 70)
    
    delayed_results = run_delayed_reward_experiment(n_jobs=-1, goal=0.0, reward_type='terminal')
    plot_delayed_reward_comparison(delayed_results, save_path='delayed_reward_comparison.png')
    print_delayed_reward_summary(delayed_results)
    
    df_delayed = pd.DataFrame({k: v["vpi"] for k, v in delayed_results.items()})
    df_delayed.to_csv('delayed_reward_vpi.csv', index=False)

    # ========================================
    # Part 5: Challenging Delayed Reward (Bellman Justification)
    # ========================================
    print("\n" + "=" * 70)
    print("PART 5: Challenging Delayed Reward (Unstable Dynamics)")
    print("=" * 70)
    
    challenging_results = run_challenging_delayed_reward(n_jobs=-1)
    plot_challenging_delayed_reward(challenging_results, save_path='challenging_delayed_reward.png')
    print_challenging_summary(challenging_results)
    
    df_challenging = pd.DataFrame({k: v["vpi"] for k, v in challenging_results.items()})
    df_challenging.to_csv('challenging_delayed_reward_vpi.csv', index=False)

    # ========================================
    # Final Comprehensive Summary
    # ========================================
    print_all_experiments_summary(
        original_results=results,
        comparative_results=comparative_results,
        tuning_results=tuning_results,
        delayed_results=delayed_results,
        challenging_results=challenging_results
    )

    # ========================================
    # List Output Files
    # ========================================
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE - OUTPUT FILES")
    print("=" * 70)
    print("""
Plots:
  - vpi_section6_1.png
  - comparative_study.png
  - bellman_tuning.png
  - delayed_reward_comparison.png
  - challenging_delayed_reward.png

Data (CSV):
  - vpi_section6_1.csv, arms_section6_1.csv
  - comparative_vpi.csv, comparative_arms.csv
  - bellman_tuning_vpi.csv
  - delayed_reward_vpi.csv
  - challenging_delayed_reward_vpi.csv
""")
    print("=" * 70)