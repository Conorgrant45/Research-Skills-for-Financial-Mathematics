import numpy as np
import math
from scipy.interpolate import interp1d

class BellmanSolver:
    """
    Generalized Bellman Solver for Diffusion Processes.
    Perfectly matched to the conor.py environment physics and boundaries.
    """
    def __init__(self, 
                 reward_fn, 
                 theta_0=0.05, 
                 theta_x=-0.1,    # The missing stabilizing gravity
                 theta_a=0.01, 
                 sigma=0.1, 
                 Delta=1.0,
                 domain_lo=-50.0, # The missing boundary walls
                 domain_hi=50.0,
                 epLen=10, 
                 action_lo=-5.0, 
                 action_hi=5.0,
                 initial_q=1837.1, 
                 n_quadrature=32):
        
        self.reward_fn = reward_fn
        self.theta_0 = theta_0
        self.theta_x = theta_x
        self.theta_a = theta_a
        self.sigma = sigma
        self.Delta = Delta
        self.sqrt_Delta = math.sqrt(Delta)
        
        self.domain_lo = domain_lo
        self.domain_hi = domain_hi
        self.epLen = epLen
        self.initial_q = initial_q
        
        self.action_lo = action_lo
        self.action_hi = action_hi

        # Gauss-Hermite Quadrature for Normal Distribution (The Noise)
        pts, wts = np.polynomial.hermite.hermgauss(n_quadrature)
        self.quad_z = pts * np.sqrt(2)
        self.quad_w = wts / np.sqrt(np.pi)

    def _get_next_state(self, x, a, z):
        """Transition dynamics exactly matching conor.py"""
        # x is scalar, a is an array of actions, z is a scalar noise weight
        drift = self.theta_0 + (self.theta_x * x) + (self.theta_a * a)
        diffusion = self.sigma * self.sqrt_Delta * z
        new_x = x + (drift * self.Delta) + diffusion
        
        # The padded room: Prevent the Oracle from flying off to infinity
        return np.clip(new_x, self.domain_lo, self.domain_hi)

    def _get_expected_reward(self, s, actions):
        """Safely passes the 1D arrays that conor.py expects"""
        rewards = np.zeros(len(actions))
        for i, a in enumerate(actions):
            # conor.py does x, a = state[0], action[0], so we must pass arrays
            rewards[i] = self.reward_fn(np.array([s], dtype=np.float64), 
                                        np.array([a], dtype=np.float64))
        return rewards

    def solve(self, n_states=500, n_actions=500):
        """Solves the Bellman equation backwards."""
        # Grid over the *actual* domain boundaries so the Oracle maps the walls
        states = np.linspace(self.domain_lo, self.domain_hi, n_states) 
        actions = np.linspace(self.action_lo, self.action_hi, n_actions)
        
        V = np.zeros((self.epLen + 1, n_states))
        self.policy = np.zeros((self.epLen, n_states))
        
        for h in range(self.epLen - 1, -1, -1):
            V_next_interp = interp1d(states, V[h + 1, :], kind='linear', fill_value="extrapolate")
            
            for i, s in enumerate(states):
                Q_sa = np.zeros(n_actions)
                
                # Get the base reward for all actions from this state
                r = self._get_expected_reward(s, actions)
                
                for z, w in zip(self.quad_z, self.quad_w):
                    s_next = self._get_next_state(s, actions, z)
                    expected_next_V = V_next_interp(s_next)
                    # Integrate expectation
                    Q_sa += w * (r + expected_next_V)
                
                best_action_idx = np.argmax(Q_sa) 
                V[h, i] = Q_sa[best_action_idx]
                self.policy[h, i] = actions[best_action_idx]
                
        self.V_table = V
        self.state_grid = states

    def get_value(self, x, h=0):
        """Fetches the exact Bellman value for the starting state."""
        final_interp = interp1d(self.state_grid, self.V_table[h, :], kind='linear', fill_value="extrapolate")
        return float(final_interp(x))