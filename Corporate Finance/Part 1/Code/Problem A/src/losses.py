"""
Bellman-AiO loss (Maliar et al.) for deep-learning solution of Eq. 3.10 in Strebulaev.

This loss function enforces two conditions simultaneously:

1) Bellman identity:
       V(k, z) = e(k, z, I) + BETA  * E[V(k', z')]
   where k' is chosen by the policy network.

2) First-order condition (FOC) for optimal investment:
       psi_I(I, k) + 1 = BETA  * E[V_k(k', z')]
   ensuring the max operator in the Bellman equation is satisfied.

We use the ALL-IN-ONE (AiO) estimator as in Maliar et al.,
which replaces two nested expectations with one expectation over
independent shock draws, producing the RV1*RV2 and RF1*RF2 structure.
"""

import tensorflow as tf
from src.model import profit, adj_cost, adj_cost_I
from src.config import BETA, DELTA, RHO_Z, SIGMA_EPS



def bellman_aio_loss(policy_net, value_net, logk, logz, eps1, eps2, lambda_foc=1.0):

    # 1. Convert log-states to levels
    k = tf.exp(logk)      # current capital level
    z = tf.exp(logz)      # current productivity level

    # Policy network: predicts dlog k
    inputs = tf.stack([logk, logz], axis=1)
    delta_log_k = tf.squeeze(policy_net(inputs), axis=1)
    
    # Next-period capital in log form and level
    logk_next = logk + delta_log_k
    k_next = tf.exp(logk_next)

    # Investment implied by the policy: I = k' − (1-DELTA)k
    I = k_next - (1.0 - DELTA) * k

    # Two independent shock draws for AiO expectation
    logz1 = RHO_Z * logz + SIGMA_EPS * eps1
    logz2 = RHO_Z * logz + SIGMA_EPS * eps2

    # Current flow payoff: profits − adjustment cost − investment
    e = profit(k, z) - adj_cost(I, k) - I
    
    # Current value function evaluation
    V = tf.squeeze(value_net(inputs), axis=1)

    # Compute V(k', z') and delV/delk' using gradient tape
    # We differentiate w.r.t. k_next (not logk_next), because the FOC
    # requires the derivative of V with respect to capital.

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(k_next)

        inputs1 = tf.stack([tf.math.log(k_next), logz1], axis=1)
        inputs2 = tf.stack([tf.math.log(k_next), logz2], axis=1)

        V1 = tf.squeeze(value_net(inputs1), axis=1)    # V(k', z1)
        V2 = tf.squeeze(value_net(inputs2), axis=1)    # V(k', z2)

    # Derivatives: delV/delk' evaluated at each shocked next-state
    dV1_dk = tape.gradient(V1, k_next)          # dV/dk' at (k', z1)
    dV2_dk = tape.gradient(V2, k_next)          # dV/dk' at (k', z2)

    del tape             # free persistent gradient tape

    # FOC residuals:
    #     -psi_I(I,k) - 1 + BETA E[ V_k(k',z') ]
    # AiO replaces expectation by product RF1*RF2 in the loss.
    # ----------------------------------------------------
    psi_I_val = adj_cost_I(I, k)              # derivative of adjustment cost wrt I
    RF1 = -psi_I_val - 1.0 + BETA * dV1_dk
    RF2 = -psi_I_val - 1.0 + BETA * dV2_dk

     # Bellman residuals:
    #     V(k,z) − [ e(k,z,I) + BETA V(k',z') ]
    # AiO again uses product RV1*RV2.
    RV1 = V - (e + BETA * V1)
    RV2 = V - (e + BETA * V2)

    # AiO combined per-sample objective:
    #       Loss_i = RV1_i * RV2_i + LAMBDA * RF1_i * RF2_i
    per_sample_loss = RV1 * RV2 + lambda_foc * RF1 * RF2

    # Average loss over batch
    loss = tf.reduce_mean(per_sample_loss)
    return loss

