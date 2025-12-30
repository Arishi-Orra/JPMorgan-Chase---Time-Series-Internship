"""
Testing module: Bellman residuals and lifetime reward.

1) compute_bellman_residuals:
       Evaluates the Bellman AiO residual RV1 * RV2 on a large
       out-of-sample test set. A consistent model should have
       small residuals centered near zero.

2) compute_lifetime_reward:
       Simulates forward under the learned policy to compute
       discounted cumulative flow payoffs.
"""

import tensorflow as tf
import numpy as np
from src.utils import sample_batch

from src.config import (
    ALPHA,
    DELTA,
    BETA,
    RHO_Z,
    SIGMA_EPS,
    LOGK_MIN,
    LOGK_MAX,
    SIGMA_LOGZ,
    PHI_ADJ
)

from src.model import profit, adj_cost, adj_cost_I


# Compute Bellman residuals for testing
def compute_bellman_residuals(policy_net, value_net, N_test=5000):
    """
    Evaluates the Bellman AiO residual:
        RV1 * RV2     where RVi = V - [ e + BETA V(k', zi) ]
    """
    
    # Sample random state pairs (k,z) and independent shocks
    logk, logz, eps1, eps2 = sample_batch(N_test)
    k = tf.exp(logk)
    z = tf.exp(logz)
    
    # Policy evaluation: compute next-period capital
    inputs = tf.stack([logk, logz], axis=1)
    delta_log_k = tf.squeeze(policy_net(inputs), axis=1)

    logk_next = logk + delta_log_k
    k_next = tf.exp(logk_next)
    
    # Investment and flow payoff
    I = k_next - (1 - DELTA)*k
    e = profit(k,z) - adj_cost(I,k) - I
    
    # Shocked next-period productivities
    logz1 = RHO_Z*logz + SIGMA_EPS*eps1
    logz2 = RHO_Z*logz + SIGMA_EPS*eps2
    
    # Compute value at current and next states
    V  = tf.squeeze(value_net(inputs), axis=1)
    # Evaluate V at (k', z1) and (k', z2)
    V1 = tf.squeeze(value_net(tf.stack([logk_next, logz1], axis=1)), axis=1)
    V2 = tf.squeeze(value_net(tf.stack([logk_next, logz2], axis=1)), axis=1)
    
    # Bellman residuals
    RB1 = V - (e + BETA*V1)
    RB2 = V - (e + BETA*V2)
    
    # AiO objective multiplies the two residuals
    return (RB1*RB2).numpy()


# Compute lifetime reward under learned policy
def compute_lifetime_reward(policy_net, value_net, N_paths=5, T=20):
    """
    Simulates forward under the policy to compute discounted sum
    of flow payoffs:

         LR = Sum BETA^t e(k_t, z_t)

    Inputs:
        N_paths : number of independent simulation paths
        T       : horizon length

    Returns:
        mean_LR       : average lifetime reward across paths
        LR_list       : all path rewards
        cum_path      : cumulative reward trajectory for the first path
    """
    
    LR_list = []
    cum_path = None         # stores time path of cumulative rewards for first path

    for p in range(N_paths):
        # Initialize log capital and log productivity
        logk = tf.random.uniform([], LOGK_MIN, LOGK_MAX)
        logz = tf.random.normal([], 0.0, SIGMA_LOGZ)

        k = tf.exp(logk)
        z = tf.exp(logz)

        LR = 0.0
        beta_pow = 1.0          # accumulator for BETA^t
        
        # Only store the full trajectory for the first path
        if p == 0:
            cum_rewards = []
            
        # Forward simulation for T steps
        for t in range(T):
            # Current-period profit
            pi = z*k**ALPHA
            
            # Policy evaluation: get dlogk --> k'
            inputs = tf.stack([[tf.math.log(k), tf.math.log(z)]])
            delta_log_k = policy_net(inputs)[0,0]
            k_next = tf.exp(tf.math.log(k) + delta_log_k)  
            
            # Investment and adjustment cost
            I = k_next - (1-DELTA)*k
            psi = 0.5*PHI_ADJ*(I*I)/(k+1e-12)
            
            # Flow payoff
            e = pi - psi - I
            
            # Accumulate discounted reward
            LR += beta_pow * float(e.numpy())
            beta_pow *= BETA

            # Store cumulative reward for first path
            if p == 0:
                cum_rewards.append(LR)
            
            # Productivity shock and capital transition
            eps = tf.random.normal([])
            z = tf.exp(RHO_Z*tf.math.log(z) + SIGMA_EPS*eps)
            k = k_next
            
        # Store path result
        LR_list.append(LR)

        if p == 0:
            cum_path = np.array(cum_rewards)

    return np.mean(LR_list), np.array(LR_list), cum_path
