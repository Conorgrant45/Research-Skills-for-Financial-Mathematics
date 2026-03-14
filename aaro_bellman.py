#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bellman Solver for the Adaptive Diffusion Portfolio Problem.

Solves V_h(x) exactly (up to discretization) via backward induction,
then compares against the RL agent's learned values.

Dynamics:
    X' = X * [1 + (theta + kappa*sum(a))*Delta + sigma*sqrt(Delta)*(a . W)]
Terminal reward:
    R_H(x) = (10 - x) * x
Out-of-bounds:
    reward = -initial_q, episode terminates
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm


class BellmanSolver:

    def __init__(self, theta=0.05, kappa=0.1, sigma=0.2, Delta=1/52,
                 rho=10, epLen=30, action_dim=2, initial_q=25,
                 n_state_grid=401, n_action_per_dim=21, n_quadrature=64):
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.Delta = Delta
        self.sqrt_Delta = math.sqrt(Delta)
        self.rho = rho
        self.epLen = epLen
        self.action_dim = action_dim
        self.initial_q = initial_q
        self.n_state_grid = n_state_grid
        self.n_quadrature = n_quadrature

        self.state_grid = np.linspace(-rho, rho, n_state_grid)
        self.action_grid = self._generate_simplex_grid(n_action_per_dim)
        self.n_actions = len(self.action_grid)

        self.quad_z, self.quad_w = self._setup_quadrature(n_quadrature)

        # Precompute action properties
        self.action_sums = np.array([np.sum(a) for a in self.action_grid])
        self.action_norms = np.array([np.linalg.norm(a) for a in self.action_grid])

        # Value function V[h, i] and optimal policy
        self.V = np.zeros((epLen, n_state_grid))
        self.policy_idx = np.zeros((epLen, n_state_grid), dtype=int)
        self.solved = False

    def _generate_simplex_grid(self, n_per_dim):
        """
        Generate grid over simplex {a : a_i >= 0, sum(a_i) <= 1}.
        For action_dim=2: triangular grid over (a1, a2).
        """
        grid_vals = np.linspace(0, 1, n_per_dim)
        actions = []

        if self.action_dim == 2:
            for a1 in grid_vals:
                for a2 in grid_vals:
                    if a1 + a2 <= 1.0 + 1e-10:
                        actions.append(np.array([a1, a2]))
        elif self.action_dim == 1:
            for a1 in grid_vals:
                actions.append(np.array([a1]))
        else:
            # For higher dimensions: strategic sampling
            actions.append(np.zeros(self.action_dim))
            for i in range(self.action_dim):
                for w in np.linspace(0.1, 1.0, 10):
                    a = np.zeros(self.action_dim)
                    a[i] = w
                    actions.append(a)
            for total in np.linspace(0.1, 1.0, 10):
                a = np.full(self.action_dim, total / self.action_dim)
                actions.append(a)
            np.random.seed(999)
            for _ in range(500):
                raw = np.random.exponential(1, self.action_dim)
                raw = raw / np.sum(raw) * np.random.uniform(0, 1)
                actions.append(raw)

        return actions

    def _setup_quadrature(self, n_points):
        """Gauss-Hermite quadrature adapted for standard normal."""
        pts, wts = np.polynomial.hermite.hermgauss(n_points)
        z = pts * np.sqrt(2)
        w = wts / np.sqrt(np.pi)
        return z, w

    def _compute_expected_value(self, x, action_idx, V_next):
        """
        Compute E[V_{h+1}(X') | X_h = x, a] using quadrature.
        
        X' = x * (1 + drift*Delta + sigma*||a||*sqrt(Delta)*Z)
        
        If |X'| > rho: value = -initial_q (penalty, episode ends)
        If |X'| <= rho: value = V_next(X') via interpolation
        """
        a_sum = self.action_sums[action_idx]
        a_norm = self.action_norms[action_idx]

        drift_factor = 1.0 + (self.theta + self.kappa * a_sum) * self.Delta
        vol_factor = self.sigma * a_norm * self.sqrt_Delta

        expected = 0.0
        for k in range(self.n_quadrature):
            z = self.quad_z[k]
            w = self.quad_w[k]

            x_next = x * (drift_factor + vol_factor * z)

            if abs(x_next) > self.rho:
                val = -self.initial_q
            else:
                val = np.interp(x_next, self.state_grid, V_next)

            expected += w * val

        return expected

    def _compute_terminal_value(self, x, action_idx):
        """
        Compute E[R_H(X') | X_{H-1}=x, a] where R(x) = (10-x)*x.
        Out of bounds: R = -initial_q.
        """
        a_sum = self.action_sums[action_idx]
        a_norm = self.action_norms[action_idx]

        drift_factor = 1.0 + (self.theta + self.kappa * a_sum) * self.Delta
        vol_factor = self.sigma * a_norm * self.sqrt_Delta

        expected = 0.0
        for k in range(self.n_quadrature):
            z = self.quad_z[k]
            w = self.quad_w[k]

            x_next = x * (drift_factor + vol_factor * z)

            if abs(x_next) > self.rho:
                val = -self.initial_q
            else:
                val = (10.0 - x_next) * x_next

            expected += w * val

        return expected

    def solve(self, verbose=True):
        """Backward induction to compute V_h(x) for all h, x."""
        if verbose:
            print("=" * 60)
            print("BELLMAN SOLVER")
            print("=" * 60)
            print(f"  State grid:   {self.n_state_grid} points on "
                  f"[{-self.rho}, {self.rho}]")
            print(f"  Action grid:  {self.n_actions} simplex points")
            print(f"  Quadrature:   {self.n_quadrature} Gauss-Hermite points")
            print(f"  Horizon:      {self.epLen} steps")
            print(f"  Solving...")

        final_step = self.epLen - 1

        # Step h = H-1 (final step): V_{H-1}(x) = max_a E[R(X')]
        if verbose:
            print(f"  h = {final_step} (terminal)...", end=" ", flush=True)

        for i, x in enumerate(self.state_grid):
            best_val = -np.inf
            best_a = 0
            for a_idx in range(self.n_actions):
                val = self._compute_terminal_value(x, a_idx)
                if val > best_val:
                    best_val = val
                    best_a = a_idx
            self.V[final_step, i] = best_val
            self.policy_idx[final_step, i] = best_a

        if verbose:
            print("done")

        # Steps h = H-2 down to 0
        for h in range(final_step - 1, -1, -1):
            if verbose:
                print(f"  h = {h}...", end=" ", flush=True)

            V_next = self.V[h + 1, :]

            for i, x in enumerate(self.state_grid):
                best_val = -np.inf
                best_a = 0
                for a_idx in range(self.n_actions):
                    val = self._compute_expected_value(x, a_idx, V_next)
                    if val > best_val:
                        best_val = val
                        best_a = a_idx
                self.V[h, i] = best_val
                self.policy_idx[h, i] = best_a

            if verbose:
                print("done")

        self.solved = True
        if verbose:
            v_at_x0 = self.get_value(2.0, 0)
            print(f"\n  V*(x0=2, h=0) = {v_at_x0:.6f}")
            print("=" * 60)

    def get_value(self, x, h):
        """Interpolate V_h(x)."""
        return np.interp(x, self.state_grid, self.V[h, :])

    def get_optimal_action(self, x, h):
        """Look up optimal action at (x, h)."""
        idx = np.argmin(np.abs(self.state_grid - x))
        return self.action_grid[self.policy_idx[h, idx]]

    def get_optimal_total_allocation(self, x, h):
        """Look up optimal sum(a*) at (x, h)."""
        a = self.get_optimal_action(x, h)
        return np.sum(a)

    def simulate_optimal_policy(self, x0, n_sims=100000, seed=123):
        """Forward simulate the solved optimal policy."""
        assert self.solved, "Call solve() first"
        rng = np.random.RandomState(seed)

        rewards = np.zeros(n_sims)
        trajectories = np.zeros((n_sims, self.epLen + 1))
        trajectories[:, 0] = x0

        for sim in range(n_sims):
            x = x0
            for h in range(self.epLen):
                a = self.get_optimal_action(x, h)
                a_sum = np.sum(a)
                a_norm = np.linalg.norm(a)

                z = rng.randn()
                drift = 1.0 + (self.theta + self.kappa * a_sum) * self.Delta
                vol = self.sigma * a_norm * self.sqrt_Delta
                x_next = x * (drift + vol * z)

                if abs(x_next) > self.rho:
                    rewards[sim] = -self.initial_q
                    trajectories[sim, h + 1:] = np.clip(x_next, -self.rho, self.rho)
                    break
                elif h == self.epLen - 1:
                    rewards[sim] = (10.0 - x_next) * x_next

                trajectories[sim, h + 1] = x_next
                x = x_next

        mean_r = np.mean(rewards)
        se_r = np.std(rewards) / np.sqrt(n_sims)
        return mean_r, se_r, trajectories

    # =========================================================================
    # Plotting methods
    # =========================================================================

    def plot_value_functions(self):
        """Plot V_h(x) at several timesteps."""
        assert self.solved
        timesteps = [0, 5, 10, 15, 20, 25, 29]
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = cm.viridis(np.linspace(0, 1, len(timesteps)))
        for t, c in zip(timesteps, colors):
            ax.plot(self.state_grid, self.V[t, :],
                    color=c, linewidth=1.5, label=f'h = {t}')

        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('State x', fontsize=12)
        ax.set_ylabel(r'$V_h(x)$', fontsize=12)
        ax.set_title('Bellman Value Function at Various Timesteps', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    def plot_value_function_zoomed(self, x_range=(0, 6)):
        """Plot V_h(x) zoomed to region where agent actually operates."""
        assert self.solved
        mask = (self.state_grid >= x_range[0]) & (self.state_grid <= x_range[1])
        x_zoom = self.state_grid[mask]

        timesteps = [0, 10, 20, 29]
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = cm.viridis(np.linspace(0, 1, len(timesteps)))
        for t, c in zip(timesteps, colors):
            ax.plot(x_zoom, self.V[t, mask],
                    color=c, linewidth=2, label=f'h = {t}')

        terminal_reward = (10.0 - x_zoom) * x_zoom
        ax.plot(x_zoom, terminal_reward, 'r--', linewidth=1.5,
                label=r'$R(x) = (10-x)x$', alpha=0.7)

        ax.axvline(x=2.0, color='gray', linestyle=':', alpha=0.6, label='x₀ = 2')
        ax.set_xlabel('State x', fontsize=12)
        ax.set_ylabel(r'$V_h(x)$', fontsize=12)
        ax.set_title('Value Function (Zoomed to Operating Region)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    def plot_optimal_policy(self):
        """Plot optimal total allocation sum(a*) as function of state."""
        assert self.solved
        timesteps = [0, 10, 20, 29]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        x_range_mask = ((self.state_grid >= 0) & (self.state_grid <= 8))
        x_plot = self.state_grid[x_range_mask]

        for idx, h in enumerate(timesteps):
            ax = axes[idx]

            total_alloc = np.zeros(len(x_plot))
            a1_alloc = np.zeros(len(x_plot))
            a2_alloc = np.zeros(len(x_plot))

            for j, x in enumerate(x_plot):
                a = self.get_optimal_action(x, h)
                total_alloc[j] = np.sum(a)
                a1_alloc[j] = a[0]
                if self.action_dim > 1:
                    a2_alloc[j] = a[1]

            ax.plot(x_plot, total_alloc, 'b-', linewidth=2,
                    label=r'$\sum a_i^*$')
            ax.plot(x_plot, a1_alloc, 'r--', linewidth=1.5,
                    label=r'$a_1^*$', alpha=0.7)
            if self.action_dim > 1:
                ax.plot(x_plot, a2_alloc, 'g--', linewidth=1.5,
                        label=r'$a_2^*$', alpha=0.7)

            ax.axvline(x=2.0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=5.0, color='orange', linestyle=':', alpha=0.5,
                       label='x = 5 (R peak)')
            ax.set_xlabel('State x', fontsize=11)
            ax.set_ylabel('Allocation', fontsize=11)
            ax.set_title(f'Optimal Policy at h = {h}', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_ylim([-0.05, 1.05])

        plt.suptitle('Optimal Policy: Action Allocations vs State', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_trajectory_distribution(self, x0=2.0, n_sims=5000):
        """Show distribution of wealth trajectories under optimal policy."""
        assert self.solved
        _, _, trajectories = self.simulate_optimal_policy(x0, n_sims, seed=42)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: sample trajectories
        ax = axes[0]
        t_axis = np.arange(self.epLen + 1)
        for i in range(min(200, n_sims)):
            ax.plot(t_axis, trajectories[i, :], alpha=0.05, color='blue',
                    linewidth=0.5)

        percentiles = [10, 25, 50, 75, 90]
        pct_vals = np.percentile(trajectories, percentiles, axis=0)
        colors_pct = ['red', 'orange', 'black', 'orange', 'red']
        styles = ['--', '--', '-', '--', '--']

        for p, pv, c, s in zip(percentiles, pct_vals, colors_pct, styles):
            ax.plot(t_axis, pv, color=c, linewidth=1.5, linestyle=s,
                    label=f'{p}th pctl')

        ax.axhline(y=5.0, color='green', linestyle=':', alpha=0.5,
                   label='x=5 (R peak)')
        ax.set_xlabel('Timestep h', fontsize=12)
        ax.set_ylabel('State x', fontsize=12)
        ax.set_title('Wealth Trajectories Under Optimal Policy', fontsize=13)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)

        # Right: terminal state distribution
        ax = axes[1]
        terminal_x = trajectories[:, -1]
        terminal_r = (10.0 - terminal_x) * terminal_x

        ax.hist(terminal_r, bins=80, density=True, alpha=0.7,
                color='tab:blue', edgecolor='white')
        ax.axvline(x=np.mean(terminal_r), color='red', linewidth=2,
                   label=f'Mean = {np.mean(terminal_r):.2f}')
        ax.set_xlabel('Terminal Reward (10-x)x', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of Terminal Rewards', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        plt.show()

    def plot_policy_heatmap(self):
        """Heatmap of optimal total allocation over (h, x)."""
        assert self.solved
        x_mask = (self.state_grid >= 0) & (self.state_grid <= 8)
        x_sub = self.state_grid[x_mask]

        alloc_map = np.zeros((self.epLen, len(x_sub)))
        for h in range(self.epLen):
            for j, x in enumerate(x_sub):
                alloc_map[h, j] = self.get_optimal_total_allocation(x, h)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(alloc_map, aspect='auto', origin='lower',
                       extent=[x_sub[0], x_sub[-1], 0, self.epLen - 1],
                       cmap='RdYlBu_r', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, label=r'Total Allocation $\sum a_i^*$')
        ax.axvline(x=2.0, color='white', linestyle='--', linewidth=1.5,
                   label='x₀ = 2')
        ax.axvline(x=5.0, color='white', linestyle=':', linewidth=1.5,
                   label='x = 5')
        ax.set_xlabel('State x', fontsize=12)
        ax.set_ylabel('Timestep h', fontsize=12)
        ax.set_title('Optimal Policy Heatmap: Total Risky Allocation', fontsize=14)
        ax.legend(fontsize=10, loc='upper right')
        plt.tight_layout()
        plt.show()