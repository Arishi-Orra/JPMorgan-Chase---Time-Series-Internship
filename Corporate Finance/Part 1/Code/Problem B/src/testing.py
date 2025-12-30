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
from src.config import (ALPHA, DELTA, BETA, RHO_Z, SIGMA_EPS, SIGMA_LOGZ)
from src.model import profit, adj_cost


# Compute Bellman residuals for testing
def compute_bellman_residuals(policy_net, value_net, N_test=5000):
    """
    Evaluates the Bellman AiO residual:
        RV1 * RV2    where: RVi = V(k, b, z) − [ e(k, b, z) + BETA * V(k', b', z_i') ]
    """

    # Sample random states (log k, b, log z) and independent shocks
    logk, b, logz, eps1, eps2 = sample_batch(N_test)
    
    # Convert states to levels
    k = tf.exp(logk)
    z = tf.exp(logz)
    
    # Policy evaluation
    inputs = tf.stack([logk, b, logz], axis=1)
    policy_out = policy_net(inputs)
    
    # Policy outputs: dlog(k) and next-period debt b'
    delta_logk = policy_out[:,0]
    b_next = policy_out[:,1]

    logk_next = logk + delta_logk
    k_next = tf.exp(logk_next)

    # Investment and flow payoff
    I = k_next - (1.0 - DELTA) * k
    e = profit(k,z) - adj_cost(I,k) - I
    
    # Shocked next-period productivities
    logz1 = RHO_Z*logz + SIGMA_EPS*eps1
    logz2 = RHO_Z*logz + SIGMA_EPS*eps2

    # Compute value at current and next states
    V = tf.squeeze(value_net(inputs), axis=1)
    # Continuation values at shocked next states
    V1 = tf.squeeze(value_net(tf.stack([logk_next,b_next,logz1],axis=1)),axis=1)
    V2 = tf.squeeze(value_net(tf.stack([logk_next,b_next,logz2],axis=1)),axis=1)
    
    # Bellman residuals
    RB1 = V - (e + BETA*V1)
    RB2 = V - (e + BETA*V2)
    
    # AiO objective multiplies the two residuals
    return (RB1 * RB2).numpy()


# Compute lifetime reward under learned policy
def compute_lifetime_reward(policy_net, ValueNet, N_paths=20, T=30):
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

    rewards = []
    cum_path = None         # stores time path of cumulative rewards for first path

    for p in range(N_paths):
        # Initialize states
        logk = tf.random.uniform([], -1.0, 2.0)
        logz = tf.random.normal([], 0.0, SIGMA_LOGZ)
        b = 0.0                  # initial debt                     

        k = tf.exp(logk)
        z = tf.exp(logz)

        LR = 0.0
        beta_pow = 1.0              # accumulator for BETA^t
        
        # Only store the full trajectory for the first path
        if p == 0:
            cum_rewards = []
            
        # Forward simulation for T steps
        for _ in range(T):
            # Policy evaluation: (dlog k, b')
            inputs = tf.stack([[tf.math.log(k), b, tf.math.log(z)]])
            policy_out = policy_net(inputs)

            delta_logk = policy_out[0,0]
            b = policy_out[0,1]
            
            # Capital transition
            k_next = tf.exp(tf.math.log(k) + delta_logk)
            
            # Investment and flow payoff
            I = k_next - (1.0 - DELTA) * k
            e = z*k**ALPHA - adj_cost(I,k) - I
            
            # Accumulate discounted reward
            LR += beta_pow * float(e.numpy())
            beta_pow *= BETA
            
            # Store cumulative reward for first path
            if p == 0:
                cum_rewards.append(LR)
                
            # Productivity shock and state transition
            eps = tf.random.normal([])
            z = tf.exp(RHO_Z*tf.math.log(z) + SIGMA_EPS*eps)
            k = k_next
            
        # Store path result
        rewards.append(LR)
        
        if p == 0:
            cum_path = np.array(cum_rewards)

    return np.mean(rewards), np.array(rewards), cum_path
