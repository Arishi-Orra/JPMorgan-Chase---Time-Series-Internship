import numpy as np

# ----------------------------------------
# Economic model parameters
# ----------------------------------------

ALPHA = 0.33      # output elasticity of capital: pi(k,z) = z * k^ALPHA
DELTA = 0.10      # depreciation rate
BETA  = 0.96      # discount factor (1 / (1 + r))

# Productivity process: ln z_{t+1} = rho * ln z_t + sigma * eps
RHO_Z      = 0.9
SIGMA_EPS  = 0.2

# Adjustment cost: psi(I, k) = 0.5 * PHI * (I^2 / k)
PHI_ADJ = 2.0

# Stationary std dev of log z for AR(1)
SIGMA_LOGZ = SIGMA_EPS / np.sqrt(1.0 - RHO_Z ** 2)

# Range for log(k) sampling 
LOGK_MIN = np.log(0.1)
LOGK_MAX = np.log(50.0)