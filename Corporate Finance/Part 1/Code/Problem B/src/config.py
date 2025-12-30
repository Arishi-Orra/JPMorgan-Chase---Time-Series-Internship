import numpy as np

# ----------------------------------------
# Economic model parameters
# ----------------------------------------

BETA = 0.96                             # discount factor: beta = (1 / (1 + r))
RISK_FREE_R = (1.0 / BETA) - 1.0        # implied risk-free interest rate 
ALPHA = 0.33                            # output elasticity of capital: pi(k,z) = z * k^ALPHA
DELTA = 0.10                            # depreciation rate

# ============================================================
# Corporate finance parameters 
# ============================================================

TAU = 0.25                  # corporate tax rate
ALPHA_BK = 0.30             # deadweight bankruptcy cost

# Productivity process: ln z_{t+1} = rho * ln z_t + sigma * eps
RHO_Z      = 0.9
SIGMA_EPS  = 0.2

# Stationary standard deviation of log(z)
SIGMA_LOGZ = SIGMA_EPS / np.sqrt(1.0 - RHO_Z**2)

# Adjustment cost: psi(I, k) = 0.5 * PHI * (I^2 / k)
PHI_ADJ = 2.0

# ============================================================
# Sampling ranges for training
# ============================================================

# log(k) sampling 
LOGK_MIN = np.log(0.1)
LOGK_MAX = np.log(50.0)

# Debt range 
B_MIN = -5.0                # negative = cash
B_MAX = 10.0                # positive = debt