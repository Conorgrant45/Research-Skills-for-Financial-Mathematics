import numpy as np
import math

class LiquidationBellman:
    def __init__(self, epLen=30, S0=100.0, price_kappa=6.0, sigma_price=4.0, 
                 mean_start=100.0, mean_end=95.0, eta_temp=20, 
                 theta_terminal=120.0, n_state_grid=200, n_quadrature=32):
        self.epLen = epLen
        self.Delta = 1/30
        self.S0, self.price_kappa, self.sigma_price = S0, price_kappa, sigma_price
        self.mean_start, self.mean_end = mean_start, mean_end
        self.eta_temp, self.theta_terminal = eta_temp, theta_terminal
        
        # State: Remaining Inventory Q in [0, 1]
        self.state_grid = np.linspace(0, 1, n_state_grid)
        # Action: Fraction a in [0, 1]
        self.action_grid = np.linspace(0, 1, 101) 
        
        # Quadrature for price shocks
        pts, wts = np.polynomial.hermite.hermgauss(n_quadrature)
        self.quad_z = pts * np.sqrt(2)
        self.quad_w = wts / np.sqrt(np.pi)
        
        self.V = np.zeros((epLen + 1, n_state_grid))
        self.policy = np.zeros((epLen, n_state_grid))

    def solve(self):
        # Terminal Value: V_H(Q) = -theta * Q^2
        self.V[self.epLen, :] = -self.theta_terminal * (self.state_grid**2)
        
        for h in range(self.epLen - 1, -1, -1):
            m_t = self.mean_start + (h / (self.epLen-1)) * (self.mean_end - self.mean_start)
            
            for i, q in enumerate(self.state_grid):
                best_val = -np.inf
                
                for a in self.action_grid:
                    sold_qty = a * q
                    q_next = q - sold_qty
                    
                    # Expected Reward calculation (simplified for linear price impact)
                    # Note: Full stochasticity is handled via the expected baseline price
                    # dS_t = kappa(m-S)dt + sigma*dW
                    expected_S_next = self.S0 # Simplified: Current price expectation
                    exec_price = max(m_t - self.eta_temp * sold_qty, 0)
                    reward = exec_price * sold_qty
                    
                    future_v = np.interp(q_next, self.state_grid, self.V[h+1, :])
                    total_v = reward + future_v
                    
                    if total_v > best_val:
                        best_val = total_v
                        self.policy[h, i] = a
                self.V[h, i] = best_val