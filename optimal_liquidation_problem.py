#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive partition RL for an optimal liquidation problem.

Key features:
- 1D state for the RL agent: remaining inventory Q_t
- Exogenous Ornstein-Uhlenbeck price process with declining mean level
- Temporary market impact
- Terminal penalty for leftover inventory
- Records:
    * episode reward
    * number of balls
    * number of splits
    * action path
    * inventory path
    * sold quantity path
    * baseline price path
    * execution price path
    * reward decomposition
    * terminal inventory
- Produces plots for:
    * reward vs episode for different scaling values
    * number of balls vs episode for different scaling values
    * runtime vs scaling
    * average action by timestep
    * average inventory by timestep
    * average sold quantity by timestep
    * average baseline/execution price by timestep
    * average reward components by timestep
    * terminal inventory distribution
    * splits per episode
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
from joblib import Parallel, delayed


# =============================================================================
# Core parameters
# =============================================================================

epLen = 30
nEps = 2000
numIters = 1

# State for RL agent: remaining inventory only
starting_state = 1.0
Delta = 1 / 30
action_dim = 1

# Price process: OU with declining mean price
S0 = 100.0
price_kappa = 6.0          # mean-reversion speed
sigma_price = 4.0          # high volatility
mean_start = 100.0
mean_end = 95.0            # declining long-run mean over the horizon

# Liquidation economics
eta_temp = 200             # temporary impact coefficient
theta_terminal = 120.0     # penalty for leftover inventory

# Keep these names to minimise code changes elsewhere
theta = 0.0
kappa = 0.0
sigma = 0.0

# Adaptive discretization hyperparameters
initial_q = 200.0
rho = 1.0                  # inventory in [0,1]
rho_1 = 0.5                # action radius around [0,1]
lip = 1.0
split_threshold = 2
scaling = 0.01

# Scaling grid for experiments
scaling_vec = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]


# =============================================================================
# Base classes
# =============================================================================

class Agent(object):
    def __init__(self):
        pass

    def update_obs(self, obs, action, reward, newObs):
        pass

    def update_policy(self, h):
        pass

    def pick_action(self, obs):
        pass

    def get_num_arms(self):
        pass


class Environment(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        return 0, 0, 0


# =============================================================================
# Liquidation environment
# =============================================================================

class AdaDiffEnvironment(Environment):
    """
    Optimal liquidation environment.

    RL state:
        Q_t = remaining inventory in [0,1]

    Hidden exogenous price:
        dS_t = kappa_price (m_t - S_t) dt + sigma_price dW_t
        where m_t declines linearly from mean_start to mean_end

    Action:
        a_t in [0,1], interpreted as fraction of remaining inventory sold

    Quantity sold:
        x_t = a_t * Q_t

    Inventory dynamics:
        Q_{t+1} = Q_t - x_t

    Execution price:
        P^exec_t = max(S_t - eta_temp * x_t, 0)

    Reward:
        r_t = P^exec_t * x_t

    Terminal penalty:
        r_T <- r_T - theta_terminal * Q_T^2
    """

    def __init__(self, epLen, starting_state):
        self.epLen = epLen
        self.starting_state = starting_state
        self.state = starting_state
        self.timestep = 0

        # hidden price state
        self.price = S0

        # current-step diagnostics, updated inside advance()
        self.last_info = {}

    def get_epLen(self):
        return self.epLen

    def declining_mean(self, t):
        frac = t / max(self.epLen - 1, 1)
        return mean_start + frac * (mean_end - mean_start)

    def reset(self):
        self.timestep = 0
        self.state = self.starting_state
        self.price = S0
        self.last_info = {}

    def advance(self, action):
        q_t = float(self.state)

        # scalar action in [0,1]
        a = float(np.clip(np.sum(action), 0.0, 1.0))

        # quantity sold
        sold_qty = a * q_t
        q_next = max(q_t - sold_qty, 0.0)

        # OU price evolution with declining mean
        mean_t = self.declining_mean(self.timestep)
        z = np.random.randn()
        price_next = (
            self.price
            + price_kappa * (mean_t - self.price) * Delta
            + sigma_price * np.sqrt(Delta) * z
        )

        # use current evolved baseline price for execution
        baseline_price = price_next
        exec_price = max(baseline_price - eta_temp * sold_qty, 0.0)

        sale_proceeds = exec_price * sold_qty
        terminal_penalty = 0.0

        reward = sale_proceeds

        if self.timestep == self.epLen - 1:
            terminal_penalty = theta_terminal * (q_next ** 2)
            reward -= terminal_penalty

        self.state = q_next
        self.price = price_next
        self.timestep += 1

        pContinue = 1
        if self.timestep == self.epLen:
            pContinue = 0

        self.last_info = {
            "inventory_before": q_t,
            "inventory_after": q_next,
            "action": a,
            "sold_qty": sold_qty,
            "baseline_price": baseline_price,
            "execution_price": exec_price,
            "sale_proceeds": sale_proceeds,
            "terminal_penalty": terminal_penalty,
            "reward": reward,
            "mean_price_level": mean_t,
        }

        return reward, q_next, pContinue


# =============================================================================
# Experiment wrapper
# =============================================================================

class Experiment(object):
    def __init__(self, env, agent_list, dictionary):
        assert isinstance(env, Environment)

        self.seed = dictionary['seed']
        self.epFreq = dictionary['recFreq']
        self.targetPath = dictionary['targetPath']
        self.deBug = dictionary['deBug']
        self.nEps = dictionary['nEps']
        self.env = env
        self.epLen = env.get_epLen()
        self.num_iters = dictionary['numIters']
        self.agent_list = agent_list

        self.data = np.zeros([dictionary['nEps'] * self.num_iters, 5])
        self.step_records = []

        np.random.seed(self.seed)

    def run(self):
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')

        for i in range(self.num_iters):
            agent = self.agent_list[i]

            for ep in range(1, self.nEps + 1):
                self.env.reset()
                agent.start_episode()

                oldState = self.env.state
                epReward = 0.0
                agent.update_policy(ep)
                pContinue = 1
                h = 0

                while pContinue > 0 and h < self.env.epLen:
                    action = agent.pick_action(oldState, h)
                    reward, newState, pContinue = self.env.advance(action)
                    epReward += reward

                    # record step-level diagnostics
                    info = self.env.last_info.copy()
                    info["episode"] = ep - 1
                    info["iteration"] = i
                    info["timestep"] = h
                    self.step_records.append(info)

                    agent.update_obs(oldState, action, reward, newState, h)
                    oldState = newState
                    h += 1

                index = i * self.nEps + (ep - 1)
                self.data[index, 0] = ep - 1
                self.data[index, 1] = i
                self.data[index, 2] = epReward
                self.data[index, 3] = agent.get_num_arms()
                self.data[index, 4] = agent.current_episode_splits

        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')

    def save_data(self):
        dt = pd.DataFrame(
            self.data,
            columns=['episode', 'iteration', 'epReward', 'Number_of_Balls', 'Splits']
        )
        dt = dt[(dt.T != 0).any()]
        return dt

    def save_step_data(self):
        return pd.DataFrame(self.step_records)


# =============================================================================
# Tree / node classes
# =============================================================================

class Node():
    def __init__(self, qVal, rEst, muEst, sigmaEst, num_visits,
                 num_unique_visits, num_splits, state_val, action_val,
                 radius, action_radius):
        self.qVal = qVal
        self.rEst = rEst
        self.muEst = muEst
        self.sigmaEst = sigmaEst
        self.num_visits = num_visits
        self.num_unique_visits = num_unique_visits
        self.num_splits = num_splits
        self.state_val = state_val
        self.action_val = np.array(action_val)
        self.radius = radius
        self.action_radius = action_radius
        self.children = None

    def split_node(self, flag, epLen):
        action_offsets = list(itertools.product([-1, 1], repeat=action_dim))
        state_offsets = [-1, 1]

        children = []

        for s_off in state_offsets:
            new_state = self.state_val + s_off * self.radius / 2

            for a_offs in action_offsets:
                new_action = self.action_val + np.array(a_offs) * self.action_radius / 2

                if self.num_visits <= 1:
                    child = Node(
                        initial_q, 0, 0, 0, self.num_visits, 0,
                        self.num_splits + 1, new_state, new_action,
                        self.radius / 2, self.action_radius / 2
                    )
                else:
                    child = Node(
                        self.qVal, self.rEst, self.muEst, self.sigmaEst,
                        self.num_visits, self.num_visits,
                        self.num_splits + 1, new_state, new_action,
                        self.radius / 2, self.action_radius / 2
                    )
                children.append(child)

        self.children = children
        return self.children


class Tree():
    def __init__(self, epLen, flag):
        self.head = Node(
            initial_q, 0, 0, 0, 0, 0, 0,
            0, np.repeat(0.5, action_dim), rho, rho_1
        )
        self.epLen = epLen
        self.flag = flag
        self.state_leaves = [self.head.state_val]
        self.vEst = [initial_q]
        self.tree_leaves = [self.head]

    def get_head(self):
        return self.head

    def split_node(self, node, timestep, previous_tree):
        children = node.split_node(self.flag, self.epLen)

        if node in self.tree_leaves:
            self.tree_leaves.remove(node)

        for child in children:
            self.tree_leaves.append(child)

        child_1_state = children[0].state_val
        child_1_radius = children[0].radius

        if np.min(np.abs(np.asarray(self.state_leaves) - child_1_state)) >= child_1_radius:
            parent = node.state_val
            parent_index = self.state_leaves.index(parent)
            parent_vEst = self.vEst[parent_index]

            self.state_leaves.pop(parent_index)
            self.vEst.pop(parent_index)

            num_action_offsets = 2 ** action_dim
            self.state_leaves.append(children[0].state_val)
            self.state_leaves.append(children[num_action_offsets].state_val)
            self.vEst.append(parent_vEst)
            self.vEst.append(parent_vEst)

        return children

    def get_num_balls(self, node):
        if node.children is None:
            return 1
        num_balls = 0
        for child in node.children:
            num_balls += self.get_num_balls(child)
        return num_balls

    def get_number_of_active_balls(self):
        return self.get_num_balls(self.head)

    def get_active_ball_recursion(self, state, node):
        if node.children is None:
            return node, node.qVal

        active_node = None
        qVal = -np.inf

        for child in node.children:
            if self.state_within_node(state, child):
                new_node, new_qVal = self.get_active_ball_recursion(state, child)
                if new_qVal >= qVal:
                    active_node, qVal = new_node, new_qVal

        if active_node is None:
            return node, node.qVal

        return active_node, qVal

    def get_active_ball(self, state):
        return self.get_active_ball_recursion(state, self.head)

    def state_within_node(self, state, node):
        return np.abs(state - node.state_val) <= node.radius


# =============================================================================
# Adaptive model-based discretization agent
# =============================================================================

class AdaptiveModelBasedDiscretization(Agent):
    def __init__(self, epLen, numIters, scaling, split_threshold, inherit_flag, flag):
        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling
        self.split_threshold = split_threshold
        self.inherit_flag = inherit_flag
        self.flag = flag
        self.tree_list = []
        self.current_episode_splits = 0

        for _ in range(epLen):
            tree = Tree(epLen, self.inherit_flag)
            self.tree_list.append(tree)

    def start_episode(self):
        self.current_episode_splits = 0

    def reset(self):
        self.tree_list = []
        self.current_episode_splits = 0
        for _ in range(self.epLen):
            tree = Tree(self.epLen, self.inherit_flag)
            self.tree_list.append(tree)

    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, action, reward, newObs, timestep):
        tree = self.tree_list[timestep]
        active_node, _ = tree.get_active_ball(obs)

        active_node.num_visits += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits

        active_node.rEst = ((t - 1) * active_node.rEst + reward) / t

        if timestep != self.epLen - 1:
            active_node.muEst = ((t - 1) * active_node.muEst + newObs - obs) / t
            active_node.sigmaEst = ((t - 1) * active_node.sigmaEst + (newObs - obs - active_node.muEst) ** 2) / t

        if self.flag is False:
            if timestep == self.epLen - 1:
                active_node.qVal = min(
                    active_node.qVal, initial_q,
                    active_node.rEst
                    + self.scaling / np.sqrt(active_node.num_visits)
                    + self.scaling * active_node.radius
                )
            else:
                next_tree = self.tree_list[timestep + 1]
                vEst = min(next_tree.vEst) + lip * (1 + active_node.muEst ** 2 + active_node.sigmaEst ** 2)
                active_node.qVal = min(
                    active_node.qVal, initial_q,
                    active_node.rEst + vEst
                    + self.scaling / np.sqrt(active_node.num_visits)
                    + self.scaling * active_node.radius
                )

            index = 0
            for state_val in tree.state_leaves:
                _, qMax = tree.get_active_ball(state_val)
                tree.vEst[index] = min(qMax, initial_q, tree.vEst[index])
                index += 1

        if active_node.children is None and t >= 2 ** (self.split_threshold * active_node.num_splits):
            if timestep >= 1:
                _ = tree.split_node(active_node, timestep, self.tree_list[timestep - 1])
            else:
                _ = tree.split_node(active_node, timestep, None)
            self.current_episode_splits += 1

    def update_policy(self, k):
        if self.flag:
            for h in np.arange(self.epLen - 1, -1, -1):
                tree = self.tree_list[h]
                for node in tree.tree_leaves:
                    if node.num_unique_visits == 0:
                        node.qVal = initial_q
                    else:
                        if h == self.epLen - 1:
                            node.qVal = min(
                                node.qVal, initial_q,
                                node.rEst + self.scaling / np.sqrt(node.num_visits)
                            )
                        else:
                            next_tree = self.tree_list[h + 1]
                            vEst = min(next_tree.vEst) + lip * (1 + node.muEst ** 2 + node.sigmaEst ** 2)
                            node.qVal = min(
                                node.qVal, initial_q,
                                node.rEst + vEst + self.scaling / np.sqrt(node.num_visits)
                            )

                index = 0
                for state_val in tree.state_leaves:
                    _, qMax = tree.get_active_ball(state_val)
                    tree.vEst[index] = min(qMax, initial_q, tree.vEst[index])
                    index += 1

    def greedy(self, state, timestep, epsilon=0):
        tree = self.tree_list[timestep]
        active_node, _ = tree.get_active_ball(state)

        action = np.zeros(action_dim)
        for i in range(action_dim):
            action[i] = np.clip(
                np.random.uniform(
                    active_node.action_val[i] - active_node.action_radius,
                    active_node.action_val[i] + active_node.action_radius
                ),
                0.0, 1.0
            )
        return action

    def pick_action(self, state, timestep):
        return self.greedy(state, timestep)



#%%
# =============================================================================
# Helper to run one experiment
# =============================================================================

def run_single_experiment(seed, scaling_value):
    env = AdaDiffEnvironment(epLen, starting_state)
    agent = AdaptiveModelBasedDiscretization(
        epLen, nEps, scaling_value, split_threshold, False, False
    )

    settings = {
        'seed': seed,
        'epFreq': 1,
        'targetPath': f'./tmp_iter_{seed}.csv',
        'deBug': False,
        'nEps': nEps,
        'recFreq': 10,
        'numIters': 1
    }

    exp = Experiment(env, [agent], settings)
    exp.run()

    summary_df = exp.save_data()
    step_df = exp.save_step_data()
    return summary_df, step_df


# =============================================================================
# Main
# =============================================================================
param_text = (
    f"epLen={epLen}, nEps={nEps}, "
    f"price_kappa={price_kappa}, sigma_price={sigma_price}, "
    f"mean_start={mean_start}, mean_end={mean_end}, "
    f"eta_temp={eta_temp}, terminal_penalty={theta_terminal}, "
    f"initial_q={initial_q}, rho={rho}, rho_1={rho_1}, "
    f"lip={lip}, split_threshold={split_threshold}"
)

if __name__ == "__main__":
    

    
    # -------------------------------------------------------------------------
    # 1. Single run for strategy plots
    # -------------------------------------------------------------------------
    summary_df, step_df = run_single_experiment(seed=0, scaling_value=0.01)

    # Reward vs episode
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['episode'], summary_df['epReward'])
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title(f"Reward vs episode\n{param_text}")    
    plt.grid(True)
    plt.show()

    # Number of balls vs episode
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['episode'], summary_df['Number_of_Balls'])
    plt.xlabel("Episode")
    plt.ylabel("Number of balls")
    plt.title(f"Number of balls vs episode\n{param_text}")
    plt.grid(True)
    plt.show()

    # Splits per episode
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df['episode'], summary_df['Splits'])
    plt.xlabel("Episode")
    plt.ylabel("Splits in episode")
    plt.title(f"Splits per episode\n{param_text}")
    plt.grid(True)
    plt.show()

    # Average action by timestep
    avg_action = step_df.groupby('timestep')['action'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_action.index, avg_action.values, marker='o')
    plt.xlabel("Timestep")
    plt.ylabel("Average action")
    plt.title(f"Average liquidation action by timestep\n{param_text}")
    plt.grid(True)
    plt.show()

    # Average inventory by timestep
    avg_inventory = step_df.groupby('timestep')['inventory_before'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_inventory.index, avg_inventory.values, marker='o')
    plt.xlabel("Timestep")
    plt.ylabel("Average remaining inventory")
    plt.title(f"Average inventory by timestep\n{param_text}")
    plt.grid(True)
    plt.show()

    # Average sold quantity by timestep
    avg_sold = step_df.groupby('timestep')['sold_qty'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_sold.index, avg_sold.values, marker='o')
    plt.xlabel("Timestep")
    plt.ylabel("Average sold quantity")
    plt.title(f"Average sold quantity by timestep\n{param_text}")
    plt.grid(True)
    plt.show()

    # Average baseline and execution price by timestep
    avg_prices = step_df.groupby('timestep')[['baseline_price', 'execution_price', 'mean_price_level']].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_prices.index, avg_prices['baseline_price'], label='Baseline price')
    plt.plot(avg_prices.index, avg_prices['execution_price'], label='Execution price')
    plt.plot(avg_prices.index, avg_prices['mean_price_level'], label='OU mean level', linestyle='--')
    plt.xlabel("Timestep")
    plt.ylabel("Price")
    plt.title(f"Average price paths\n{param_text}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Average reward components by timestep
    avg_rewards = step_df.groupby('timestep')[['sale_proceeds', 'terminal_penalty', 'reward']].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards.index, avg_rewards['sale_proceeds'], label='Sale proceeds')
    plt.plot(avg_rewards.index, avg_rewards['terminal_penalty'], label='Terminal penalty')
    plt.plot(avg_rewards.index, avg_rewards['reward'], label='Net reward')
    plt.xlabel("Timestep")
    plt.ylabel("Average value")
    plt.title(f"Average reward decomposition by timestep\n{param_text}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Terminal inventory distribution
    terminal_inventory = step_df[step_df['timestep'] == epLen - 1]['inventory_after']

    plt.figure(figsize=(10, 6))
    plt.hist(terminal_inventory, bins=30)
    plt.xlabel("Terminal inventory")
    plt.ylabel("Count")
    plt.title(f"Terminal inventory distribution\n{param_text}")
    plt.grid(True)
    plt.show()

    # Action vs inventory scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(step_df['inventory_before'], step_df['action'], alpha=0.2)
    plt.xlabel("Inventory before trade")
    plt.ylabel("Action")
    plt.title("Action vs inventory\n{param_text}")
    plt.grid(True)
    plt.show()

    # -------------------------------------------------------------------------
    # 2. Scaling comparison plots
    # -------------------------------------------------------------------------
    n_replications = 5

    reward_curves = {}
    balls_curves = {}
    runtime_records = []

    for scaling_value in scaling_vec:
        t0 = time.time()

        results = Parallel(n_jobs=-1)(
            delayed(run_single_experiment)(seed=i, scaling_value=scaling_value)
            for i in range(n_replications)
        )

        summary_list = [r[0] for r in results]

        reward_df = pd.DataFrame([df['epReward'].values for df in summary_list]).T
        balls_df = pd.DataFrame([df['Number_of_Balls'].values for df in summary_list]).T

        reward_curves[scaling_value] = reward_df.mean(axis=1)
        balls_curves[scaling_value] = balls_df.mean(axis=1)

        runtime_records.append([scaling_value, time.time() - t0])

    # Reward vs episode for different scaling values
    plt.figure(figsize=(10, 6))
    for scaling_value in scaling_vec:
        plt.plot(reward_curves[scaling_value], label=f'scaling={scaling_value}')
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("Reward vs episode for different scaling values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Number of balls vs episode for different scaling values
    plt.figure(figsize=(10, 6))
    for scaling_value in scaling_vec:
        plt.plot(balls_curves[scaling_value], label=f'scaling={scaling_value}')
    plt.xlabel("Episode")
    plt.ylabel("Average number of balls")
    plt.title("Number of balls vs episode for different scaling values")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Runtime vs scaling
    runtime_table = pd.DataFrame(runtime_records, columns=['Scaling', 'Runtime_seconds'])

    plt.figure(figsize=(8, 5))
    plt.bar(runtime_table['Scaling'].astype(str), runtime_table['Runtime_seconds'])
    plt.xlabel("Scaling parameter")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs scaling")
    plt.grid(axis='y')
    plt.show()
