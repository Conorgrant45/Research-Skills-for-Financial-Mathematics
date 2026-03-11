#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:18:56 2026

@author: kylemcgillivray

Research project
with Conor and Aaro
"""
import numpy as np
import math  # CHANGE 1: Use math.sqrt for scalar operations (faster than np.sqrt)
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from joblib import Parallel, delayed
# CHANGE 2: Removed unused imports: gymnasium, matplotlib.patches, mpl, os.path,
# shutil.copyfile, pickle, time - reduces import overhead

""""
Notes: One-step update that doesn't use the full bellman update and go a planned path is so much better
It basically takes the average ...

To fix the out-of-bounds code: to advance after new_state, add: new_state = np.clip(new_state, -rho, rho)
and to tree in get_active_ball:
                            safe_state = np.clip(state, 
                            self.head.state_val - self.head.radius, 
                            self.head.state_val + self.head.radius)
        
                            active_node, qVal = self.get_active_ball_recursion(safe_state, self.head)
                            return active_node, qVal


"""


# --- Core simulation parameters ---
epLen = 30
nEps = 2000
numIters = 1
starting_state = 2
theta = 0.05
kappa = 0.1
sigma = 0.2
Delta = 1/52
action_dim = 2

# --- Algorithm hyperparameters ---
initial_q = 50
rho = 50
rho_1 = 0.5
lip = 1
split_threshold = 2
scaling = 0.01

# Precompute constants used repeatedly in hot loops
_SQRT_DELTA = math.sqrt(Delta) 
_INV_ACTION_DIM = 1.0 / action_dim
# Precompute the action offset array once at module level (was recomputed every split)
_ACTION_OFFSETS = np.array(list(itertools.product([-1, 1], repeat=action_dim)), dtype=np.float64)
_STATE_OFFSETS = np.array([-1, 1], dtype=np.float64)
_NUM_ACTION_OFFSETS = 2 ** action_dim


class Agent(object):
    """Abstract base class for a reinforcement learning agent."""

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
    """Abstract base class for a reinforcement learning environment."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        return 0, 0, 0

class ContinuousAIGym(Environment):
    """A wrapper class to adapt an OpenAI Gym environment for use with this framework."""
    def __init__(self, env, epLen):
        self.env = env
        self.epLen = epLen
        self.timestep = 0
        self.state = self.env.reset()

    def get_epLen(self):
        return self.epLen

    def reset(self):
        self.timestep = 0
        self.state = self.env.reset()

    def advance(self, action):
        newState, reward, terminal, info = self.env.step(action)
        if self.timestep == self.epLen or terminal:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1
        return reward, newState, pContinue

class AdaDiffEnvironment(Environment):
    """A custom environment simulating an adaptive diffusion process."""
    def __init__(self, epLen, starting_state):
        self.epLen = epLen
        self.state = starting_state
        self.starting_state = starting_state
        self.timestep = 0
        self._final_step = epLen - 1

    def get_epLen(self):
        return self.epLen

    def reset(self):
        self.timestep = 0
        self.state = self.starting_state

    def advance(self, action):
        action_sum = np.sum(action)
        noise = np.random.randn(action_dim)
        state = self.state  

        new_state = state + theta * state * Delta + kappa * state * action_sum * Delta + \
                    sigma * state * _SQRT_DELTA * np.dot(action, noise)
        # Handles out-of-bound rho by forcing the new_state to be within the bounds
        new_state = np.clip(new_state, -rho, rho)

        reward = 0
        if self.timestep == self._final_step:
            reward = (10 - new_state) * new_state

        self.state = new_state
        self.timestep += 1

        pContinue = 1
        if self.timestep == self.epLen:
            pContinue = 0

        return reward, new_state, pContinue

class Experiment(object):

    def __init__(self, env, agent_list, dict):
        assert isinstance(env, Environment)

        self.seed = dict['seed']
        self.epFreq = dict['recFreq']
        self.targetPath = dict['targetPath']
        self.deBug = dict['deBug']
        self.nEps = dict['nEps']
        self.env = env
        self.epLen = env.get_epLen()
        self.num_iters = dict['numIters']
        self.agent_list = agent_list
        self.data = np.zeros([dict['nEps'] * self.num_iters, 4])
        np.random.seed(self.seed)

    def run(self):
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')

        env = self.env
        deBug = self.deBug
        epLen_local = env.epLen
        data = self.data

        for i in range(self.num_iters):
            agent = self.agent_list[i]
            for ep in range(1, nEps + 1):
                env.reset()
                oldState = env.state
                epReward = 0
                agent.update_policy(ep)
                pContinue = 1
                h = 0

                if deBug:
                    while pContinue > 0 and h < epLen_local:
                        print('state : ' + str(oldState))
                        action = agent.pick_action(oldState, h)
                        print('action : ' + str(action))
                        reward, newState, pContinue = env.advance(action)
                        epReward += reward
                        agent.update_obs(oldState, action, reward, newState, h)
                        oldState = newState
                        h += 1
                    print('final state: ' + str(newState))
                    print('Total Reward: ' + str(epReward))
                else:
                    while pContinue > 0 and h < epLen_local:
                        action = agent.pick_action(oldState, h)
                        reward, newState, pContinue = env.advance(action)
                        epReward += reward
                        agent.update_obs(oldState, action, reward, newState, h)
                        oldState = newState
                        h += 1

                index = ep - 1
                data[index, 0] = index
                data[index, 1] = i
                data[index, 2] = epReward
                data[index, 3] = agent.get_num_arms()

        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')

    def save_data(self):
       print('**************************************************')
       print('Saving data')
       print('**************************************************')
       dt = pd.DataFrame(self.data, columns=['episode', 'iteration', 'epReward', 'Number_of_Balls'])
       dt = dt[(dt.T != 0).any()]
       return dt


class Node():
    """Represents a node in the state-action space partition."""

    __slots__ = ('qVal', 'rEst', 'muEst', 'sigmaEst', 'num_visits',
                 'num_unique_visits', 'num_splits', 'state_val', 'action_val',
                 'radius', 'action_radius', 'children')

    def __init__(self, qVal, rEst, muEst, sigmaEst, num_visits, num_unique_visits,
                 num_splits, state_val, action_val, radius, action_radius):
        self.qVal = qVal
        self.rEst = rEst
        self.muEst = muEst
        self.sigmaEst = sigmaEst
        self.num_visits = num_visits
        self.num_unique_visits = num_unique_visits
        self.num_splits = num_splits
        self.state_val = state_val
        self.action_val = np.asarray(action_val, dtype=np.float64)
        self.radius = radius
        self.action_radius = action_radius
        self.children = None

    def split_node(self, flag, epLen):
        half_radius = self.radius * 0.5  # CHANGE 14: Precompute halved values
        half_action_radius = self.action_radius * 0.5
        action_val = self.action_val
        state_val = self.state_val
        num_splits_plus1 = self.num_splits + 1
        low_visits = self.num_visits <= 1

        children = []

        for s_off in _STATE_OFFSETS:
            new_state = state_val + s_off * half_radius

            for a_offs in _ACTION_OFFSETS:
                new_action = action_val + a_offs * half_action_radius

                if low_visits:
                    child = Node(initial_q, 0, 0, 0, self.num_visits, 0,
                               num_splits_plus1, new_state, new_action,
                               half_radius, half_action_radius)
                else:
                    child = Node(self.qVal, self.rEst, self.muEst, self.sigmaEst,
                               self.num_visits, self.num_visits, num_splits_plus1,
                               new_state, new_action, half_radius, half_action_radius)
                children.append(child)

        self.children = children
        return self.children


class Tree():
    """Manages the hierarchy of nodes for a specific timestep."""

    def __init__(self, epLen, flag):
        self.head = Node(initial_q, 0, 0, 0, 0, 0, 0, 0, np.full(action_dim, 0.5), rho, rho_1)
        self.epLen = epLen
        self.flag = flag
        self.state_leaves = [self.head.state_val]
        self.vEst = [initial_q]
        self.tree_leaves = [self.head]
        self._min_vEst = initial_q

    def get_head(self):
        return self.head

    def _update_min_vEst(self):
        """CHANGE 18: Helper to update cached min when vEst changes."""
        self._min_vEst = min(self.vEst)

    def split_node(self, node, timestep, previous_tree):
        children = node.split_node(self.flag, self.epLen)

        self.tree_leaves.remove(node)
        self.tree_leaves.extend(children)

        child_1_state = children[0].state_val
        child_1_radius = children[0].radius

        state_leaves = self.state_leaves
        min_dist = min(abs(sl - child_1_state) for sl in state_leaves)

        if min_dist >= child_1_radius:
            parent = node.state_val
            parent_index = state_leaves.index(parent)
            parent_vEst = self.vEst[parent_index]

            state_leaves.pop(parent_index)
            self.vEst.pop(parent_index)

            state_leaves.append(children[0].state_val)
            state_leaves.append(children[_NUM_ACTION_OFFSETS].state_val)
            self.vEst.append(parent_vEst)
            self.vEst.append(parent_vEst)
            self._update_min_vEst()

        return children

    def get_number_of_active_balls(self):
        return len(self.tree_leaves)

    def get_num_balls(self, node):
        if node.children is None:
            return 1
        num_balls = 0
        for child in node.children:
            num_balls += self.get_num_balls(child)
        return num_balls

    def get_active_ball_recursion(self, state, node):
        if node.children is None:
            return node, node.qVal

        active_node = None
        qVal = -math.inf

        for child in node.children:
            if abs(state - child.state_val) <= child.radius:
                new_node, new_qVal = self.get_active_ball_recursion(state, child)
                if new_qVal >= qVal:
                    active_node = new_node
                    qVal = new_qVal

        if active_node is None:
            return node, node.qVal

        return active_node, qVal

    def get_active_ball(self, state):
        # Handles now out-of-bound rho trees
        safe_state = np.clip(state, 
                            self.head.state_val - self.head.radius, 
                            self.head.state_val + self.head.radius)
        
        active_node, qVal = self.get_active_ball_recursion(safe_state, self.head)
        return active_node, qVal

    def state_within_node(self, state, node):
        return abs(state - node.state_val) <= node.radius


class AdaptiveModelBasedDiscretization(Agent):
    """Implements the Adaptive Model-Based Discretization agent."""

    def __init__(self, epLen, numIters, scaling, split_threshold, inherit_flag, flag):
        self.epLen = epLen
        self.numIters = numIters
        self.scaling = scaling
        self.split_threshold = split_threshold
        self.inherit_flag = inherit_flag
        self.flag = flag
        self.tree_list = [Tree(epLen, self.inherit_flag) for _ in range(epLen)]
        self._final_step = epLen - 1

    def reset(self):
        self.tree_list = [Tree(self.epLen, self.inherit_flag) for _ in range(self.epLen)]

    def get_num_arms(self):
        return sum(tree.get_number_of_active_balls() for tree in self.tree_list)

    def update_obs(self, obs, action, reward, newObs, timestep):
        tree = self.tree_list[timestep]
        active_node, _ = tree.get_active_ball(obs)

        active_node.num_visits += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits

        inv_t = 1.0 / t
        t_minus_1_ratio = (t - 1) * inv_t

        active_node.rEst = t_minus_1_ratio * active_node.rEst + reward * inv_t

        if timestep != self._final_step:
            delta_state = newObs - obs
            active_node.muEst = t_minus_1_ratio * active_node.muEst + delta_state * inv_t
            active_node.sigmaEst = t_minus_1_ratio * active_node.sigmaEst + \
                                   (delta_state - active_node.muEst) ** 2 * inv_t

        if not self.flag:
            ucb_visit = self.scaling / math.sqrt(active_node.num_visits)
            ucb_radius = self.scaling * active_node.radius

            if timestep == self._final_step:
                active_node.qVal = min(active_node.qVal, initial_q,
                                       active_node.rEst + ucb_visit + ucb_radius)
            else:
                next_tree = self.tree_list[timestep + 1]
                vEst = next_tree._min_vEst + lip * (1 + active_node.muEst ** 2 + active_node.sigmaEst ** 2)
                active_node.qVal = min(active_node.qVal, initial_q,
                                       active_node.rEst + vEst + ucb_visit + ucb_radius)

            # Update V-estimates for this tree
            vEst_list = tree.vEst
            for idx, state_val in enumerate(tree.state_leaves):
                _, qMax = tree.get_active_ball(state_val)
                vEst_list[idx] = min(qMax, initial_q, vEst_list[idx])
            tree._update_min_vEst()

        if t >= 2 ** (self.split_threshold * active_node.num_splits):
            if timestep >= 1:
                tree.split_node(active_node, timestep, self.tree_list[timestep - 1])
            else:
                tree.split_node(active_node, timestep, None)

    def update_policy(self, k):
        if self.flag:
            for h in range(self._final_step, -1, -1):
                tree = self.tree_list[h]
                for node in tree.tree_leaves:
                    if node.num_unique_visits == 0:
                        node.qVal = initial_q
                    else:
                        if h == self._final_step:
                            node.qVal = min(node.qVal, initial_q,
                                          node.rEst + self.scaling / math.sqrt(node.num_visits))
                        else:
                            next_tree = self.tree_list[h + 1]
                            vEst = next_tree._min_vEst + lip * (1 + node.muEst ** 2 + node.sigmaEst ** 2)
                            node.qVal = min(node.qVal, initial_q,
                                          node.rEst + vEst + self.scaling / math.sqrt(node.num_visits))

                vEst_list = tree.vEst
                for idx, state_val in enumerate(tree.state_leaves):
                    _, qMax = tree.get_active_ball(state_val)
                    vEst_list[idx] = min(qMax, initial_q, vEst_list[idx])
                tree._update_min_vEst()

    def split_ball(self, node):
        children = node.split_ball()
        return children

    def greedy(self, state, timestep, epsilon=0):
        tree = self.tree_list[timestep]
        active_node, _ = tree.get_active_ball(state)

        action = _INV_ACTION_DIM * np.random.uniform(
            active_node.action_val - active_node.action_radius,
            active_node.action_val + active_node.action_radius
        )
        return action

    def pick_action(self, state, timestep):
        action = self.greedy(state, timestep)
        return action



def make_diffMDP(epLen, starting_state):
    return AdaDiffEnvironment(epLen, starting_state)

def run_single_experiment_iteration(iteration_seed):
    env_single = make_diffMDP(epLen, starting_state)
    agent_single = AdaptiveModelBasedDiscretization(epLen, nEps, scaling, split_threshold, False, False)
    dictionary_single = {
        'seed': iteration_seed, 'epFreq': 1,
        'targetPath': './tmp_iter_{}.csv'.format(iteration_seed),
        'deBug': False, 'nEps': nEps, 'recFreq': 10, 'numIters': 1
    }
    exp_single = Experiment(env_single, [agent_single], dictionary_single)
    exp_single.run()
    dt_data_single = exp_single.save_data()
    # CHANGE 37: Return .values (numpy array) to avoid pickling overhead in joblib
    return dt_data_single.epReward.values

n = 50

list_of_vpi = Parallel(n_jobs=-1)(delayed(run_single_experiment_iteration)(i) for i in range(n))

vpi_df = pd.DataFrame(list_of_vpi).T
vpi_estimate = vpi_df.mean(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(range(len(vpi_estimate)), vpi_estimate, label='vpi')
plt.xlabel("Episode")
plt.ylabel("vpi")
plt.title("vpi vs episode")
plt.legend()
plt.grid(True)
plt.show()