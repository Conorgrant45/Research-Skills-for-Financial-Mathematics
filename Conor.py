import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple
from joblib import Parallel, delayed


# ── Reward functions ─────────────────────────────────────────────────────────

def reward_6_1(state: np.ndarray, action: np.ndarray) -> float:
    """R ~ N(-(x-a)^2, 0.01). Baseline from Section 6.1. a*(x)=x."""
    x, a = state[0], action[0]
    diff = x - a
    return -(diff * diff) + np.random.normal(0.0, 0.01)


def reward_quadratic_asymmetric(state: np.ndarray, action: np.ndarray) -> float:
    """
    R ~ N(-(x-a)^2 * (1+|x|), 0.01).
    a*(x) = x. Growth order m=1.
    Sub-optimality gap grows with |x|, creating spatially
    non-uniform near-optimal set geometry for c sensitivity analysis.
    """
    x, a = state[0], action[0]
    diff = x - a
    return -(diff ** 2) * (1.0 + abs(x)) + np.random.normal(0.0, 0.01)


def reward_quadratic_shifted(state: np.ndarray, action: np.ndarray) -> float:
    """
    R ~ N(-(x-a-0.5)^2, 0.01).
    a*(x) = x - 0.5. Growth order m=1.
    Same geometry as baseline but non-trivial optimal policy.
    Control experiment: c sensitivity should match baseline asymptotically.
    """
    x, a = state[0], action[0]
    diff = x - a - 0.5
    return -(diff ** 2) + np.random.normal(0.0, 0.01)


def reward_quartic(state: np.ndarray, action: np.ndarray) -> float:
    """
    R ~ N(-(x-a)^4, 0.01).
    a*(x) = x. Growth order m=3.
    Flat well near optimum inflates zmax,c.
    Regret exponent is most sensitive to c of all four rewards.
    """
    x, a = state[0], action[0]
    diff = x - a
    return -(diff ** 4) + np.random.normal(0.0, 0.01)


# ── Configuration ─────────────────────────────────────────────────────────────

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
    # Action space bounds — set symmetrically for new rewards
    action_lo: float = -5.0
    action_hi: float = 5.0
    initial_q: float = 1837.1
    rho: float = 10.0
    rho_1: float = 5.0          # action radius = (action_hi - action_lo) / 2
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
    _sigma_sqrt_delta: float = field(init=False, repr=False)
    _action_center: float = field(init=False, repr=False)

    def __post_init__(self):
        self._sigma_sqrt_delta = self.sigma * math.sqrt(self.delta)
        self._action_center = (self.action_hi + self.action_lo) / 2.0
        # Ensure rho_1 is consistent with action bounds
        self.rho_1 = (self.action_hi - self.action_lo) / 2.0


# ── Environment ───────────────────────────────────────────────────────────────

class Agent:
    __slots__ = ()
    def update_obs(self, obs, action, reward, newObs, timestep): pass
    def update_policy(self, k): pass
    def pick_action(self, obs, timestep): pass
    def get_num_arms(self): pass


class Environment:
    __slots__ = ()
    def reset(self): pass
    def advance(self, action): return 0, 0, 0
    def get_epLen(self): return 0


class LinearDiffEnvironment(Environment):
    """
    Section 6.1 environment with linear diffusion dynamics.
    X_{h+1} = X_h + (theta_0 + theta_x*X_h + theta_a*a_h)*Delta
                   + sigma*sqrt(Delta)*Z_h
    """
    __slots__ = ('cfg', 'epLen', '_start', 'state', 'timestep',
                 'projection_count', '_sigma_sqrt_delta',
                 '_domain_lo', '_domain_hi')

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
        cfg = self.cfg
        x, a = self.state[0], action[0]

        drift = cfg.theta_0 + cfg.theta_x * x + cfg.theta_a * a
        noise = np.random.randn()
        new_x = x + drift * cfg.delta + self._sigma_sqrt_delta * noise

        new_x = min(max(new_x, self._domain_lo), self._domain_hi)
        if new_x != x + drift * cfg.delta + self._sigma_sqrt_delta * noise:
            self.projection_count += 1

        self.state[0] = new_x
        reward = cfg.reward_step_fn(np.array([x], dtype=np.float64), action)
        self.timestep += 1
        pContinue = 1 if self.timestep < self.epLen else 0
        return reward, self.state, pContinue


# ── Tree / Node ───────────────────────────────────────────────────────────────

class Node:
    __slots__ = (
        'qVal', 'rEst', 'muEst', 'sigmaEst',
        'num_visits', 'num_unique_visits', 'num_splits',
        'state_val', 'action_val', 'radius', 'action_radius',
        'children', '_state_center', '_action_center'
    )

    def __init__(self, qVal, rEst, muEst, sigmaEst,
                 num_visits, num_unique_visits, num_splits,
                 state_val, action_val, radius, action_radius):
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
        return abs(state_scalar - self._state_center) <= self.radius

    def contains(self, state: np.ndarray) -> bool:
        return abs(state[0] - self._state_center) <= self.radius

    def split_node_1d(self, initial_q: float) -> List['Node']:
        half_r = self.radius * 0.5
        half_ar = self.action_radius * 0.5
        sc = self._state_center
        ac = self._action_center

        inherit = self.num_visits > 1
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
        children = []

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
    __slots__ = ('cfg', 'initial_q', 'head', 'tree_leaves',
                 'state_leaves', 'vEst', '_state_to_idx', 'min_vEst',
                 '_lookup_cache', '_vEst_dirty')

    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.initial_q = cfg.initial_q

        # Centre of state space at 0, action space at cfg._action_center
        start_state = np.array([0.0], dtype=np.float64)
        start_action = np.array([cfg._action_center], dtype=np.float64)

        self.head = Node(
            cfg.initial_q, 0.0,
            np.zeros(1, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
            0, 0, 0,
            start_state, start_action,
            cfg.rho,   # state radius
            cfg.rho_1  # action radius = (action_hi - action_lo) / 2
        )

        self.tree_leaves: Dict[int, Node] = {id(self.head): self.head}
        self.state_leaves: List[float] = [0.0]
        self.vEst: List[float] = [float(cfg.initial_q)]
        self._state_to_idx: Dict[float, int] = {0.0: 0}
        self.min_vEst: float = float(cfg.initial_q)
        self._lookup_cache: Dict[float, Node] = {}
        self._vEst_dirty = False

    def get_active_ball_1d(self, state_scalar: float) -> Tuple[Node, float]:
        lo = self.head._state_center - self.head.radius
        hi = self.head._state_center + self.head.radius
        safe_state = min(max(state_scalar, lo), hi)

        cached = self._lookup_cache.get(safe_state)
        if cached is not None and cached.children is None:
            return cached, cached.qVal

        node = self.head
        while node.children is not None:
            best_node = None
            best_q = -math.inf
            for child in node.children:
                if child.contains_1d(safe_state):
                    if child.qVal >= best_q:
                        best_q = child.qVal
                        best_node = child
            if best_node is None:
                break
            node = best_node

        self._lookup_cache[safe_state] = node
        return node, node.qVal

    def get_active_ball(self, state: np.ndarray) -> Tuple[Node, float]:
        return self.get_active_ball_1d(state[0])

    def split_node_1d(self, node: Node) -> List[Node]:
        children = node.split_node_1d(self.initial_q)
        del self.tree_leaves[id(node)]
        for child in children:
            self.tree_leaves[id(child)] = child
        self._lookup_cache.clear()

        child_0_center = children[0]._state_center
        child_0_radius = children[0].radius

        needs_new_states = not any(
            abs(sc - child_0_center) < child_0_radius
            for sc in self.state_leaves
        )

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
        return len(self.tree_leaves)


# ── Agent ─────────────────────────────────────────────────────────────────────

class AdaptiveModelBasedDiscretization(Agent):
    __slots__ = ('cfg', 'epLen', 'scaling', 'alpha', 'split_threshold', 'lip',
                 'initial_q', 'state_dim', 'tree_list', '_split_thresholds',
                 '_action_lo', '_action_hi')

    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.epLen = cfg.epLen
        self.scaling = cfg.scaling
        self.alpha = cfg.alpha
        self.split_threshold = cfg.split_threshold
        self.lip = cfg.lip
        self.initial_q = cfg.initial_q
        self.state_dim = cfg.state_dim
        # Store action bounds from config — no more hardcoded [0,10]
        self._action_lo = cfg.action_lo
        self._action_hi = cfg.action_hi

        self._split_thresholds = [
            2 ** (cfg.split_threshold * i) for i in range(21)
        ]
        self.tree_list = [Tree(cfg) for _ in range(cfg.epLen)]

    def reset(self) -> None:
        self.tree_list = [Tree(self.cfg) for _ in range(self.epLen)]

    def get_num_arms(self) -> int:
        return sum(t.get_number_of_active_balls() for t in self.tree_list)

    def update_obs(self, obs: np.ndarray, action: np.ndarray,
                   reward: float, newObs: np.ndarray, timestep: int) -> None:
        tree = self.tree_list[timestep]
        obs_scalar = obs[0]
        active_node, _ = tree.get_active_ball_1d(obs_scalar)

        active_node.num_visits += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits

        active_node.rEst = ((t - 1) * active_node.rEst + reward) / t

        is_terminal = (timestep == self.epLen - 1)
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
            next_tree = self.tree_list[timestep + 1]
            mu_sq = active_node.muEst[0] ** 2
            sigma_sq = active_node.sigmaEst[0]
            vEst_next = next_tree.min_vEst + self.lip * (1.0 + mu_sq + sigma_sq)
            q_new = active_node.rEst + vEst_next + ucb

        active_node.qVal = min(active_node.qVal, self.initial_q, q_new)
        tree.update_vEst()

        num_splits = active_node.num_splits
        threshold = (self._split_thresholds[num_splits]
                     if num_splits < len(self._split_thresholds)
                     else 2 ** (self.split_threshold * num_splits))

        if t >= threshold:
            tree.split_node_1d(active_node)

    def update_policy(self, k: int) -> None:
        pass

    def pick_action(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Select action uniformly from active node's action region,
        clipped to the configured action space bounds."""
        tree = self.tree_list[timestep]
        active_node, _ = tree.get_active_ball_1d(state[0])

        ac = active_node._action_center
        ar = active_node.action_radius

        # Use config bounds instead of hardcoded [0, 10]
        lo = max(self._action_lo, ac - ar)
        hi = min(self._action_hi, ac + ar)

        return np.array([np.random.uniform(lo, hi)], dtype=np.float64)


# ── Experiment runner ─────────────────────────────────────────────────────────

class Experiment:
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


def run_one_seed(seed: int, cfg: ExpConfig) -> Tuple[np.ndarray, np.ndarray]:
    env = LinearDiffEnvironment(cfg)
    agent = AdaptiveModelBasedDiscretization(cfg)
    exp = Experiment(env, agent, cfg, seed)
    return exp.run()


def run_experiment(configs: List[ExpConfig],
                   n_jobs: int = -1) -> Dict[str, Dict[str, np.ndarray]]:
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
            "arms": arm_matrix.mean(axis=0)
        }
        final_mean = results[cfg.label]["vpi"][-100:].mean()
        print(f"    Done. Final mean VPI: {final_mean:.3f}")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_vpi(results: Dict[str, Dict[str, np.ndarray]],
             smooth_window: int = 50,
             save_path: str = 'vpi_results.png') -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.cm.tab10(np.linspace(0, 0.8, len(results)))

    for (label, series), colour in zip(results.items(), colours):
        vpi = series["vpi"]
        episodes = np.arange(len(vpi))
        cumsum = np.cumsum(np.insert(vpi, 0, 0))
        smoothed = np.array([
            (cumsum[i + 1] - cumsum[max(0, i - smooth_window + 1)])
            / (i - max(0, i - smooth_window + 1) + 1)
            for i in range(len(vpi))
        ])
        ax.plot(episodes, smoothed, label=label, color=colour, linewidth=2)
        ax.plot(episodes, vpi, color=colour, alpha=0.12, linewidth=0.7)

    ax.set_xlabel('Episode (K)', fontsize=13)
    ax.set_ylabel('Mean VPI', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {save_path}')
    plt.close(fig)


# ── Configs ───────────────────────────────────────────────────────────────────

# Baseline: Section 6.1 — action space [0,10] preserved exactly
CFG_6_1 = ExpConfig(
    starting_state=4.0,
    action_lo=0.0,
    action_hi=10.0,
    initial_q=1837.1,
    rho=10.0,
    reward_step_fn=reward_6_1,
    label='Baseline 6.1',
)

# Reward 1: Asymmetric quadratic — symmetric action space [-5,5]
# initial_q upper bound: max state ~10, worst case -(0-10)^2*(1+10)*H = -1100*10
# Use positive upper bound ~ 0 since reward is always negative; set small positive
CFG_ASYMMETRIC = ExpConfig(
    starting_state=4.0,
    action_lo=-5.0,
    action_hi=5.0,
    initial_q=100.0,   # reward is always <= 0 so this is a safe upper bound
    rho=10.0,
    reward_step_fn=reward_quadratic_asymmetric,
    label='Asymmetric Quadratic',
)

# Reward 2: Shifted quadratic — symmetric action space [-5,5]
# Same scale as baseline so initial_q can stay similar
CFG_SHIFTED = ExpConfig(
    starting_state=4.0,
    action_lo=-5.0,
    action_hi=5.0,
    initial_q=1837.1,
    rho=10.0,
    reward_step_fn=reward_quadratic_shifted,
    label='Shifted Quadratic',
)

# Reward 3: Quartic — symmetric action space [-5,5]
# Reward scale is (x-a)^4, worst case ~10^4=10000 over H=10 steps
# initial_q should be a loose upper bound on cumulative reward
CFG_QUARTIC = ExpConfig(
    starting_state=4.0,
    action_lo=-5.0,
    action_hi=5.0,
    initial_q=100.0,   # quartic is always <=0, small positive upper bound
    rho=10.0,
    reward_step_fn=reward_quartic,
    label='Quartic',
)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    configs = [CFG_6_1, CFG_ASYMMETRIC, CFG_SHIFTED, CFG_QUARTIC]
    results = run_experiment(configs, n_jobs=-1)
    plot_vpi(results, smooth_window=50, save_path='vpi_all_rewards.png')

    for name in ('vpi', 'arms'):
        df = pd.DataFrame({k: v[name] for k, v in results.items()})
        df.index.name = 'episode'
        path = f'{name}_all_rewards.csv'
        df.to_csv(path, index=False)
        print(f'Saved {path}')