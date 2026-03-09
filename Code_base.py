#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:18:56 2026

@author: kylemcgillivray

Research project
with Conor and Aaro
"""
import numpy as np
import gym
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import os.path as path
from shutil import copyfile
import pickle
import time
import itertools

# This section defines the core simulation parameters for the reinforcement learning environment.
# These values control the length of an episode, the number of episodes, and the dynamics of the simulated process.
epLen = 30 # H: Defines the maximum number of steps or time stages within a single simulation episode.
nEps = 2000 # K: Specifies the total number of episodes (or learning iterations) the agent will undergo.
numIters =1 # The iteration number, potentially useful for future extensions or multi-agent scenarios.
starting_state=2 # The initial state value from which each episode begins.
theta=0.05 # The drift parameter in the Ornstein-Uhlenbeck (O-U) process, influencing the tendency of the state to return to a mean.
kappa=0.1 # The action parameter in the drift term, representing how the agent's action influences the state's drift.
sigma=0.2 # The volatility parameter in the O-U process, determining the magnitude of random fluctuations in the state.
Delta=1/52 # The step size or time increment for each step within an episode, often representing a discrete time unit.
action_dim = 2  # This sets the dimensionality of the action space. The agent will output a vector of this size.

# This section defines hyperparameters for the Adaptive Model-Based Discretization algorithm.
# These parameters often require tuning to achieve optimal performance and convergence.
initial_q=50 # The initial optimistic Q-value and V-value estimation used for unvisited states or nodes.
rho=50 # The initial radius for state space partitioning, defining the size of the initial 'balls' or regions.
rho_1=0.5 # The initial radius for action space partitioning, defining the size of action regions within each state ball.
lip=1 # The Lipschitz constant, used in the Bellman update to bound the growth of the value function.
split_threshold=2 # A constant that determines when a node (state-action region) should be split, based on the number of visits.
scaling=0.01 # A scaling constant for the UCB (Upper Confidence Bound) bonus term, balancing exploration and exploitation.


#%%

class Agent(object):
    """Abstract base class for a reinforcement learning agent.
    Defines the standard interface that any agent must implement to interact with the environment and learn."""

    def __init__(self):
        """Initializes the agent. Specific agent implementations will set up their internal state here."""
        pass

    def update_obs(self, obs, action, reward, newObs):
        """Abstract method to update the agent's internal records or model based on a new observation.

        Args:
            obs: The state before the action was taken.
            action: The action chosen by the agent.
            reward: The reward received after taking the action.
            newObs: The new state observed after taking the action.
        """

    def update_policy(self, h):
        """Abstract method to update the agent's decision-making policy.

        Args:
            h: The current timestep or episode number, which might influence policy updates (e.g., for episodic learning).
        """

    def pick_action(self, obs):
        """Abstract method to select an action based on the current observation.

        Args:
            obs: The current state observed by the agent.

        Returns:
            The chosen action.
        """
    def get_num_arms(self):
        """Abstract method to return the number of active 'arms' or regions currently being considered by the agent.
        This is often used in algorithms that discretize the state-action space into a growing number of regions.
        """

class Environment(object):
    """Abstract base class for a reinforcement learning environment.
    Defines the standard interface for resetting the environment and advancing it one step."""

    def __init__(self):
        """Initializes the environment. Specific environment implementations will set up their internal state here."""
        pass

    def reset(self):
        """Abstract method to reset the environment to an initial state, typically at the beginning of an episode."""
        pass

    def advance(self, action):
        """Abstract method to move the environment forward one step based on the given action.

        Args:
            action: The action taken by the agent.

        Returns:
            reward: The reward received for the transition.
            newState: The new state of the environment.
            pContinue: A flag (0 or 1) indicating if the episode continues (1) or terminates (0).
        """
        return 0, 0, 0

class ContinuousAIGym(Environment):
    """A wrapper class to adapt an OpenAI Gym environment for use with this framework,
    handling episodic termination and state management."""
    def __init__(self, env, epLen):
        """
        Initializes the ContinuousAIGym wrapper.
            env: An instance of an OpenAI Gym environment.
            epLen: The maximum number of steps allowed per episode for this environment.
        """
        self.env = env
        self.epLen = epLen
        self.timestep = 0 # Tracks the current step within an episode.
        self.state = self.env.reset() # Resets the underlying Gym environment and stores the initial state.


    def get_epLen(self):
        """Returns the maximum episode length."""
        return self.epLen

    def reset(self):
        """Resets the environment for a new episode by resetting the timestep and the underlying Gym environment."""
        self.timestep = 0
        self.state = self.env.reset()

    def advance(self, action):
        """Advances the environment by one step using the Gym environment's step function.
        It also manages episode termination based on maximum episode length or Gym's terminal state.

        Args:
        action: The action chosen by the agent.
        Returns:
            reward: The reward received.
            newState: The state after taking the action.
            pContinue: 0 if episode ends, 1 if it continues.
        """
        # Executes one step in the underlying Gym environment.
        newState, reward, terminal, info = self.env.step(action)

        # Checks if the episode has reached its maximum length or if the Gym environment signals termination.
        if self.timestep == self.epLen or terminal:
            pContinue = 0 # Episode ends.
            self.reset() # Resets the environment for the next episode.
        else:
            pContinue = 1 # Episode continues.

        return reward, newState, pContinue

class AdaDiffEnvironment(Environment):
    """A custom environment simulating an adaptive diffusion process,
    where the state evolves based on drift, action, and stochastic volatility."""
    def __init__(self, epLen, starting_state):
        """
        Initializes the Adaptive Diffusion Environment.
        epLen: The maximum number of steps per episode.
        starting_state: The initial state for each episode.
        """
        self.epLen = epLen
        self.state = starting_state # Current state of the process.
        self.starting_state = starting_state # Stores the fixed starting state.
        self.timestep = 0 # Tracks the current step within an episode.


    def get_epLen(self):
        """Returns the maximum episode length."""
        return self.epLen

    def reset(self):
        """Resets the environment to its starting state for a new episode."""
        self.timestep = 0
        self.state = self.starting_state

    def advance(self, action):
        """Simulates one step of the adaptive diffusion process.
        The state evolves based on an Ornstein-Uhlenbeck-like dynamic, influenced by the action.

        Args:
            action: A multi-dimensional action vector from the agent.

        Returns:
            reward: The reward received for this step.
            new_state: The state after the transition.
            pContinue: 0 if episode ends, 1 if it continues.
        """

        # Calculates the sum of action components to represent an aggregate effect on the drift.
        action_sum = np.sum(action)

        noise = np.random.randn(action_dim)

        # Computes the new state using a discretized Ornstein-Uhlenbeck process equation.
        # The state change depends on current state, drift (theta*state), action effect (kappa*state*action_sum),
        # and stochastic volatility (sigma*state*sqrt(Delta)*sum(action*noise)).
        new_state = self.state + theta * self.state* Delta+kappa * self.state * action_sum * Delta+ \
                    sigma * self.state * np.sqrt(Delta) * np.sum(action * noise)

        reward = 0
        # The reward is only given at the final timestep of an episode.
        if self.timestep == self.epLen - 1:
            # The reward function encourages the state to be around a value of 10.
            reward = (10 - new_state) * new_state

        self.state = new_state # Updates the environment's current state.
        self.timestep += 1 # Increments the internal timestep counter.

        pContinue = 1
        # If the episode length is reached, the episode terminates.
        if self.timestep == self.epLen:
            pContinue = 0

        return reward, new_state, pContinue

class Experiment(object):

    def __init__(self, env, agent_list, dict):
        """Initializes an experiment to run an MDP simulation with one or more agents.

        Args:
            env: An instance of an Environment (e.g., AdaDiffEnvironment).
            agent_list: A list of Agent instances to be evaluated.
            dict: A dictionary containing experiment configuration parameters like seed, frequency of recording, etc.
        """
        assert isinstance(env, Environment), "Provided environment must be an instance of the Environment class."

        self.seed = dict['seed'] # Random seed for reproducibility.
        self.epFreq = dict['recFreq'] # Frequency of episodes to record data.
        self.targetPath = dict['targetPath'] # File path for saving experimental data.
        self.deBug = dict['deBug'] # Boolean flag to enable/disable debug prints.
        self.nEps = dict['nEps'] # Total number of episodes to run.
        self.env = env # The environment instance.
        self.epLen = env.get_epLen() # Episode length obtained from the environment.
        self.num_iters = dict['numIters'] # Number of iterations (or agents) to run.
        self.agent_list = agent_list # List of agents participating in the experiment.

        # Initializes a NumPy array to store experiment results: episode number, iteration, episode reward, and number of active balls.
        self.data = np.zeros([dict['nEps']*self.num_iters, 4])

        np.random.seed(self.seed) # Sets the random seed for NumPy for consistent results.

    def run(self):
        """Executes the main experimental loop, running each agent for the specified number of episodes."""
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')
        for i in range(self.num_iters): # Loops through each agent (or iteration).
            agent = self.agent_list[i] # Selects the current agent.
            for ep in range(1, nEps+1): # Loops through each episode for the current agent.
                #print('Episode : ' + str(ep))
                self.env.reset() # Resets the environment at the start of each episode.
                oldState = self.env.state # Records the initial state of the episode.
                epReward = 0 # Initializes the cumulative reward for the current episode.
                agent.update_policy(ep) # Allows the agent to update its policy based on past experiences.
                pContinue = 1 # Flag to indicate if the episode is still ongoing.
                h = 0 # Timestep counter within the episode.
                while pContinue > 0 and h < self.env.epLen: # Continues as long as the episode is active and within length.
                    if self.deBug:
                        print('state : ' + str(oldState))
                    action = agent.pick_action(oldState, h) # Agent selects an action based on the current state and timestep.
                    if self.deBug:
                        print('action : ' + str(action))

                    reward, newState, pContinue = self.env.advance(action) # Environment advances, returning reward, new state, and continuation flag.
                    epReward += reward # Accumulates the reward for the episode.

                    agent.update_obs(oldState, action, reward, newState, h) # Agent updates its model/records with the new observation.
                    oldState = newState # Current state becomes the old state for the next step.
                    h = h + 1 # Increments timestep.
                if self.deBug:
                    print('final state: ' + str(newState))
                    print('Total Reward: ' + str(epReward))

                # Logs the results of the current episode into the data array.
                index = ep-1
                self.data[index, 0] = ep-1 # Episode number.
                self.data[index, 1] = i # Iteration number.
                self.data[index, 2] = epReward # Total reward for the episode.
                self.data[index, 3] = agent.get_num_arms() # Number of active 'balls' or regions used by the agent.

        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')

    def save_data(self):
       """Converts the collected experimental data into a pandas DataFrame and returns it.
       It filters out any rows that might be empty (all zeros)."""
       print('**************************************************')
       print('Saving data')
       print('**************************************************')

       # Creates a DataFrame from the collected data with meaningful column names.
       dt = pd.DataFrame(self.data, columns=['episode', 'iteration', 'epReward', 'Number_of_Balls'])
       # Filters out any rows where all values are zero, which might occur if nEps * numIters is larger than actual data logged.
       dt = dt[(dt.T != 0).any()]
       return dt

class Node():
    """Represents a node in the state-action space partition, essentially a 'ball' or region.
    Each node stores estimates of Q-values, rewards, and transition dynamics, along with visit counts and dimensions."""
    def __init__(self, qVal, rEst, muEst, sigmaEst, num_visits, num_unique_visits, num_splits, state_val, action_val, radius, action_radius):
        """Initializes a Node with its estimated values and properties.

        Args:
            qVal: Estimated Q-value (expected total reward from this node onwards).
            rEst: Estimated immediate reward for this node.
            muEst: Estimated mean of the state transition (change in state).
            sigmaEst: Estimated variance of the state transition.
            num_visits: Total number of times this node or its ancestors have been visited.
            num_unique_visits: Number of distinct times this node itself has been observed.
            num_splits: Number of times ancestors of this node have been split.
            state_val: The center of the state component of this node.
            action_val: The center of the action component of this node (a multi-dimensional array).
            radius: The radius of the state component of this node.
            action_radius: The radius of the action component of this node.
        """
        self.qVal = qVal
        self.rEst = rEst
        self.muEst = muEst
        self.sigmaEst=sigmaEst
        self.num_visits = num_visits
        self.num_unique_visits = num_unique_visits
        self.num_splits = num_splits
        self.state_val = state_val
        self.action_val = np.array(action_val) # Ensures action_val is a NumPy array for vectorized operations.
        self.radius = radius
        self.action_radius = action_radius
        self.children = None # Initially, a node has no children.


    def split_node(self, flag, epLen):
        """Splits the current node into a set of child nodes, effectively refining the state-action space partition.
        This method creates children by halving the radius in both state and action dimensions.

        Args:
            flag: A boolean flag (not directly used in this implementation but kept for compatibility).
            epLen: The episode length (not directly used in this implementation but kept for compatibility).

        Returns:
            A list of newly created child nodes.
        """
        # CHANGED: The splitting logic is enhanced to handle multi-dimensional action spaces.
        # Generates all binary combinations of -1 and 1 for each action dimension.
        # For action_dim=2, this creates (-1,-1), (-1,1), (1,-1), (1,1).
        action_offsets = list(itertools.product([-1, 1], repeat=action_dim))
        state_offsets = [-1, 1] # For state space, it's always two offsets (e.g., left and right).

        children = []

        # Iterates through each possible state offset to create new state centers.
        for s_off in state_offsets:
            # New state center is half a radius away from the parent's state center.
            new_state = self.state_val + s_off * self.radius / 2

            # For each state offset, iterate through all action offset combinations.
            for a_offs in action_offsets:
                # New action center is half an action_radius away from the parent's action center.
                new_action = self.action_val + np.array(a_offs) * self.action_radius / 2

                # If the node has been visited very few times, initialize child with optimistic Q-value.
                if self.num_visits <= 1:
                    child = Node(initial_q, 0, 0, 0, self.num_visits, 0,
                               self.num_splits+1, new_state, new_action,
                               self.radius/2, self.action_radius/2)
                # Otherwise, children inherit the parent's estimates.
                else:
                    child = Node(self.qVal, self.rEst, self.muEst, self.sigmaEst,
                               self.num_visits, self.num_visits, self.num_splits+1,
                               new_state, new_action, self.radius/2, self.action_radius/2)
                children.append(child)

        self.children = children # Assigns the new children to this node.
        return self.children # Returns the list of new children.

'''The tree class consists of a hierarchy of nodes, representing the multi-level state-action space partition.'''
class Tree():
    """Manages the hierarchy of nodes (balls) for a specific timestep, enabling dynamic partitioning
    and efficient lookup of active regions."""
    # Defines a tree by the number of steps for the initialization
    def __init__(self, epLen, flag):
        """Initializes the tree for a given timestep.

        Args:
            epLen: The episode length.
            flag: A boolean flag for the update type (full or one-step).
        """
        # The head of the tree is the largest, initial node covering the entire (or a large part of) state-action space.
        # It's initialized with optimistic Q-values, centered at 0 for state and 0.5 for action, with initial radii.
        self.head = Node(initial_q, 0, 0, 0, 0, 0, 0, 0, np.repeat(0.5, action_dim), rho,rho_1)
        self.epLen = epLen # Stores episode length.
        self.flag = flag # Stores the update flag.
        self.state_leaves = [self.head.state_val] # List of state centers of the active (leaf) nodes.
        self.vEst = [initial_q] # List of value function estimates corresponding to state_leaves.
        self.tree_leaves = [self.head] # List of all current leaf nodes in the tree.

    def get_head(self):
        """Returns the head (root) node of the tree."""
        return self.head

    def split_node(self, node, timestep, previous_tree):
        """Initiates the splitting of a given node into its children, and updates the tree's internal structure.

        Args:
            node: The node to be split.
            timestep: The current timestep (for context, not directly used in split logic).
            previous_tree: The tree from the previous timestep (for context, not directly used in split logic).

        Returns:
            A list of the newly created child nodes.
        """
        children = node.split_node(self.flag, self.epLen) # Calls the node's own split method.

        # Updates the list of leaf nodes: removes the parent and adds its children.
        self.tree_leaves.remove(node)
        for child in children:
            self.tree_leaves.append(child)

        # Retrieves properties of one child to check for state partition adjustments.
        child_1_state = children[0].state_val
        child_1_radius = children[0].radius

        # This condition checks if the splitting of the current node has created distinct state regions
        # that are sufficiently far apart to warrant updating the induced state partition.
        # It compares the distance between the existing state leaves and the new child's state.
        if np.min(np.abs(np.asarray(self.state_leaves) - child_1_state)) >= child_1_radius:


            parent = node.state_val

            parent_index = self.state_leaves.index(parent) # Finds the index of the parent's state in the state_leaves list.
            parent_vEst = self.vEst[parent_index] # Retrieves the parent's value estimate.

            self.state_leaves.pop(parent_index) # Removes the parent's state.
            self.vEst.pop(parent_index) # Removes the parent's value estimate.

            # Appends the two distinct state centers created by the split.
            # For a 2^action_dim * 2 split, the children with s_off=-1 are first, then s_off=1.
            num_action_offsets = 2**action_dim
            self.state_leaves.append(children[0].state_val) # Child with s_off=-1
            self.state_leaves.append(children[num_action_offsets].state_val) # Child with s_off=1
            # Copies the parent's value estimate to these new state regions.
            self.vEst.append(parent_vEst)
            self.vEst.append(parent_vEst)
            # Lastly we need to adjust the transition kernel estimates from the previous tree

            # Need to remove parent's state value from state_leaves,
            # add in the state values for the children
            # copy over the estimate of the value function
            # print(self.state_leaves)
        return children

    def get_num_balls(self, node):
        """Recursively counts the total number of 'active' (leaf) nodes under a given node.

        Args:
            node: The starting node for the count.

        Returns:
            The total number of leaf nodes in the subtree rooted at 'node'.
        """
        num_balls = 0
        if node.children == None: # If a node has no children, it's a leaf node (an active ball).
            return 1
        else: # If it has children, recursively count balls in each child's subtree.
            for child in node.children:
                num_balls += self.get_num_balls(child)
        return num_balls

    def get_number_of_active_balls(self):
        """Returns the total number of active (leaf) nodes in the entire tree.
        This is a measure of the agent's current discretization granularity."""
        return self.get_num_balls(self.head) # Starts the recursive count from the head of the tree.

    def get_active_ball_recursion(self, state, node):
        """Recursively searches for the most specific (smallest radius) node that contains the given state.
        If multiple children contain the state, it prioritizes the one with the highest Q-value.

        Args:
            state: The current state to locate within the tree.
            node: The current node being examined in the recursive search.

        Returns:
            active_node: The most specific node containing the state.
            qVal: The Q-value of the active_node.
        """
        # If the current node has no children, it is the most specific node for this path.
        if node.children == None:
           return node, node.qVal
        else:
        # Otherwise, check each child node.
            active_node = None
            qVal = -np.inf # Initialize with a very small value to ensure the first valid qVal is selected.

            for child in node.children:
                # If the child node contains the current state.
                if self.state_within_node(state, child):
                    # Recursively search within this child's subtree.
                    new_node, new_qVal = self.get_active_ball_recursion(state, child)
                    # If the Q-value from the child's subtree is higher or equal, update the active node.
                    if new_qVal >= qVal:
                        active_node, qVal = new_node, new_qVal
            # Fallback: If no child contains the state (should ideally not happen with correct splitting),
            # return the current node. This prevents errors but might indicate an issue in the splitting logic.
            if active_node is None:
                return node, node.qVal

            return active_node, qVal


    def get_active_ball(self, state):
        """Starts the recursive search for the active ball (node) containing the given state from the head of the tree."""
        active_node, qVal = self.get_active_ball_recursion(state, self.head)
        return active_node, qVal

    def state_within_node(self, state, node):
        """Helper method to check if a given state falls within the state radius of a node.
        Assumes a single-dimensional state for simplicity in this check."""
        return np.abs(state - node.state_val) <= node.radius

class AdaptiveModelBasedDiscretization(Agent):
    """Implements the Adaptive Model-Based Discretization agent, which dynamically partitions
    the state-action space to learn optimal policies. It maintains a tree-like structure of 'balls'
    or regions for each timestep."""

    def __init__(self, epLen, numIters, scaling, split_threshold, inherit_flag, flag):
        """Initializes the Adaptive Model-Based Discretization agent.

        Args:
            epLen: Number of steps per episode.
            numIters: Total number of iterations (episodes) the agent will run.
            scaling: Scaling parameter for the UCB (Upper Confidence Bound) exploration term.
            split_threshold: Constant determining when to split a node.
            inherit_flag: Boolean, if True, children nodes inherit values from parent (not fully implemented in provided code).
            flag: Boolean, if True, performs full Bellman updates at end of episode; if False, performs one-step updates.
        """

        self.epLen = epLen # Maximum steps per episode.
        self.numIters = numIters # Total iterations.
        self.scaling = scaling # UCB scaling parameter.
        self.split_threshold = split_threshold # Node splitting constant.

        self.inherit_flag = inherit_flag # Inheritance flag.
        self.flag = flag # Update type flag.
        self.tree_list = [] # A list to hold a separate Tree structure for each timestep 'h'.

        # Creates a new Tree for each possible timestep in an episode.
        for _ in range(epLen):
            tree = Tree(epLen, self.inherit_flag)
            self.tree_list.append(tree)

    def reset(self):
        '''Resets the agent's internal state, effectively clearing all learned partitions and estimates.'''
        # Re-initializes the list of trees, effectively resetting the state-action space partition.
        self.tree_list = []
        for _ in range(self.epLen):
            tree = Tree(self.epLen, self.inherit_flag)
            self.tree_list.append(tree)

    def get_num_arms(self):
        '''Calculates the total number of active 'arms' or leaf nodes across all timesteps.'''
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, action, reward, newObs, timestep):
        """Updates the agent's model (node estimates) based on a new observed transition.
        This method also triggers node splitting based on visit counts.

        Args:
            obs: The state before action.
            action: The action taken.
            reward: The immediate reward.
            newObs: The state after action.
            timestep: The current timestep within the episode.
        """
        # Selects the appropriate tree for the current timestep.
        tree = self.tree_list[timestep]

        # Finds the most specific node ('active ball') that contains the observed state.
        active_node, _ = tree.get_active_ball(obs)

        # Increments visit counters for the active node.
        active_node.num_visits += 1
        active_node.num_unique_visits += 1
        t = active_node.num_unique_visits # Shorthand for unique visits.

        # Updates the empirical estimate of the average reward for this node.
        active_node.rEst = ((t-1)*active_node.rEst + reward) / t

        # If it's not the last timestep, update estimates for the transition kernel (mean and variance of state change).
        if timestep != self.epLen - 1:
            # next_tree = self.tree_list[timestep+1] # Not directly used here, but relevant for V-value updates.
            active_node.muEst=((t-1)*active_node.muEst + newObs-obs) / t # Updates mean state change.
            active_node.sigmaEst=((t-1)*active_node.sigmaEst + (newObs-obs-active_node.muEst)**2) / t # Updates variance of state change.

        # If the 'flag' is False, perform one-step Bellman updates.
        if self.flag == False:
            if timestep == self.epLen - 1: # If it's the last timestep, Q-value is just reward + UCB bonus.
                active_node.qVal = min(active_node.qVal, initial_q, active_node.rEst + self.scaling / np.sqrt(active_node.num_visits)+self.scaling*active_node.radius)
            else: # Otherwise, Q-value includes an estimate of the next state's value (V-value) and transition model.
                next_tree = self.tree_list[timestep+1]
                # Estimates the value of the next state using the minimum V-estimate from the next tree,
                # adjusted by a Lipschitz-like term involving the estimated transition dynamics.
                vEst =min(next_tree.vEst)+ lip*(1+active_node.muEst**2+active_node.sigmaEst**2)
                # Updates Q-value using Bellman equation with optimistic initialization and UCB bonus.
                active_node.qVal = min(active_node.qVal, initial_q, active_node.rEst + vEst + self.scaling / np.sqrt(active_node.num_visits)+self.scaling*active_node.radius)

            # After updating a node's Q-value, update the overall value function estimate for the tree.
            index = 0
            for state_val in tree.state_leaves:
                _, qMax = tree.get_active_ball(state_val) # Find the max Q-value for that state.
                tree.vEst[index] = min(qMax, initial_q, tree.vEst[index]) # Update V-estimate, ensuring optimism.
                index += 1

        # The condition `t >= 2**(self.split_threshold * active_node.num_splits)` ensures that nodes are split
        # exponentially more frequently as their ancestors have been split more often, creating finer partitions.
        if t >= 2**(self.split_threshold * active_node.num_splits):
            # Calls the tree's method to split the active node.
            # The `previous_tree` argument is not used in the current `split_node` implementation but allows for future extensions.
            if timestep >= 1:
                _ = tree.split_node(active_node, timestep, self.tree_list[timestep-1])
            else:
                _ = tree.split_node(active_node, timestep, None)

    def update_policy(self, k):
        """Updates the agent's policy by solving empirical Bellman equations,
        especially when 'flag' is True (full Bellman updates).

        Args:
            k: The current episode number.
        """
        # If the 'flag' is True, performs full Bellman updates iterating backwards through timesteps.
        if self.flag:
            # Iterates from the last timestep (epLen-1) down to 0.
            for h in np.arange(self.epLen-1,-1,-1):
                # Gets the specific tree corresponding to the current timestep 'h'.
                tree = self.tree_list[h]
                for node in tree.tree_leaves: # Iterates through all leaf nodes in the current tree.
                    # If a node has not been visited, it retains its optimistic initial Q-value.
                    if node.num_unique_visits == 0:
                        node.qVal = initial_q
                    else:
                        # For visited nodes, update Q-value based on empirical estimates and UCB bonus.
                        if h == self.epLen - 1: # At the last timestep, Q-value is immediate reward plus UCB.
                            node.qVal = min(node.qVal, initial_q, node.rEst + self.scaling / np.sqrt(node.num_visits))
                        else: # For other timesteps, incorporate the estimated value of the next state.
                            next_tree = self.tree_list[h+1]
                            # Estimates the value of the next state, similar to update_obs.
                            vEst =min(next_tree.vEst)+ lip*(1+node.muEst**2+node.sigmaEst**2)
                            # Updates Q-value with immediate reward, estimated next state value, and UCB bonus.
                            node.qVal = min(node.qVal, initial_q, node.rEst + vEst + self.scaling / np.sqrt(node.num_visits))

                # After updating all Q-values for the current timestep, update the value function estimates (V-values).
                index = 0
                for state_val in tree.state_leaves:
                    _, qMax = tree.get_active_ball(state_val) # Find the best Q-value for each state region.
                    tree.vEst[index] = min(qMax, initial_q, tree.vEst[index]) # Update V-estimate, maintaining optimism.
                    index += 1
        pass # If flag is False, policy is updated in update_obs, so this method does nothing.

    def split_ball(self, node):
        """Placeholder method; node splitting is handled by the Node and Tree classes directly."""
        children = node.split_ball()
        return children

    def greedy(self, state, timestep, epsilon=0):
        """Selects an action based on a greedy policy (or epsilon-greedy if epsilon > 0).
        The agent finds the 'active ball' for the current state and picks an action uniformly
        from within that ball's action space.

        Args:
            state: The current state.
            timestep: The current timestep.
            epsilon: Epsilon value for epsilon-greedy exploration (not fully utilized in current uniform sampling).

        Returns:
            action: The chosen multi-dimensional action vector.
        """
        # Retrieves the relevant tree for the current timestep.
        tree = self.tree_list[timestep]

        # Identifies the most specific node (ball) that contains the current state.
        active_node, _ = tree.get_active_ball(state)

        # Initializes a multi-dimensional action vector.
        action = np.zeros(action_dim)
        # For each dimension of the action, samples an action uniformly within the active node's action radius.
        for i in range(action_dim):
                action[i] = (1/action_dim)*np.random.uniform(
                active_node.action_val[i] - active_node.action_radius,
                active_node.action_val[i] + active_node.action_radius
                 )
        # Note: The (1/action_dim) factor is a scaling for each component, ensuring the sum has a reasonable scale.
        # This ensures that the chosen action is within the defined boundaries of the active action ball.
        return action

    def pick_action(self, state, timestep):
        """The main method for the agent to choose an action.
        It directly calls the greedy action selection strategy.

        Args:
            state: The current state.
            timestep: The current timestep.

        Returns:
            action: The selected action.
        """
        action = self.greedy(state, timestep) # Delegates action selection to the greedy method.
        return action
    
    
    
#%%

from joblib import Parallel, delayed

# Helper function to create an instance of the Adaptive Diffusion Environment.
def make_diffMDP(epLen, starting_state):
    return AdaDiffEnvironment(epLen, starting_state)

# Defines a function to run a single iteration of the experiment.
# This function will be executed in parallel across multiple processes.
def run_single_experiment_iteration(iteration_seed):
    # Re-create the environment for each parallel iteration to ensure independent simulations.
    env_single = make_diffMDP(epLen, starting_state)

    # Re-create the agent for each iteration. This ensures each agent starts fresh
    # and is trained on its own separate environment run.
    agent_single = AdaptiveModelBasedDiscretization(epLen, nEps, scaling, split_threshold, False, False)

    # Dictionary containing configuration for this single experiment run.
    # numIters is set to 1 because each parallel job runs one agent's simulation.
    dictionary_single = {'seed': iteration_seed, 'epFreq' : 1, 'targetPath': './tmp_iter_{}.csv'.format(iteration_seed), 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : 1}

    # Initialize the Experiment class with the single environment and agent.
    exp_single = Experiment(env_single, [agent_single], dictionary_single)

    # Run the experiment simulation for this iteration.
    exp_single.run()

    # Save the data generated by this single experiment run and extract the episode rewards.
    dt_data_single = exp_single.save_data()
    return dt_data_single.epReward

n = 50 # Number of parallel runs (simulations) to average the results over.

# Uses joblib's Parallel and delayed functions to execute `run_single_experiment_iteration`
# for `n` times in parallel. `n_jobs=-1` utilizes all available CPU cores.
list_of_vpi = Parallel(n_jobs=-1)(delayed(run_single_experiment_iteration)(i) for i in range(n))

# Converts the list of episode rewards from all parallel runs into a DataFrame.
# Each column represents the rewards from one parallel experiment.
vpi_df = pd.DataFrame(list_of_vpi).T

# Calculates the mean episode reward across all parallel iterations for each episode step.
# This provides a more robust estimate of the agent's performance over time.
vpi_estimate = vpi_df.mean(axis=1)

# Plotting the Value Performance Index (VPI) over episodes.
# VPI represents the average accumulated reward per episode.
plt.figure(figsize=(10, 6))
plt.plot(range(len(vpi_estimate)), vpi_estimate, label='vpi')
plt.xlabel("Episode")
plt.ylabel("vpi")
plt.title("vpi vs episode")
plt.legend()
plt.grid(True)
plt.show()


#%%
vpi_estimate.to_csv('vpi_estimate_m.csv', index=False) # Saves the calculated VPI estimates to a CSV file.
print('vpi_estimate saved to vpi_estimate_m.csv') # Confirms the file saving to the user.
from google.colab import files # Imports the files utility from Google Colab.
files.download('vpi_estimate_m.csv') # Triggers a download of the CSV file to the user's local machine.

#%% New from here
# Ideas

