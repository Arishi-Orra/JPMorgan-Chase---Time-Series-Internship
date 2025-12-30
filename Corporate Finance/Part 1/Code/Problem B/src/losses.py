"""
Bellman-AiO loss (Maliar et al.) for deep-learning solution of
Section 3.6 in Strebulaev (2012): Risky Debt with Default.

This loss enforces the Bellman identity for the equity value of the firm,
taking into account endogenous default and risky debt pricing.

Key differences relative to Section 3.1 (no-debt model):

1) NO first-order condition (FOC) term:
       Debt choice and default introduce kinks and non-differentiabilities,
       so the max operator is handled directly instead of via FOCs.

2) Endogenous default:
       Equity value is truncated at zero:
           V = max{ 0, continuation value }

3) Endogenous risky interest rate:
       The promised debt payoff depends on default probabilities and
       expected recovery values.

As in Maliar et al., we use the ALL-IN-ONE (AiO) estimator, replacing
nested expectations with a single expectation over independent shock draws.
"""

import tensorflow as tf

from src.model import profit, adj_cost, recovery
from src.config import BETA, DELTA, RHO_Z, SIGMA_EPS, B_MAX, TAU, RISK_FREE_R


def bellman_aio_loss(policy_net, value_net, logk, b, logz, eps1, eps2):
    """
    Computes the Bellman-AiO loss:

        Loss = E[ RV1 * RV2 ]

    where:

        RVi = V(k, b, z) − max{ 0, e(k, b, z) + BETA V(k', b', z') }
    """
    # --------------------------------------------------------
    # Convert log-states to levels
    # --------------------------------------------------------

    k = tf.exp(logk)
    z = tf.exp(logz)

    # --------------------------------------------------------
    # Policy evaluation
    # --------------------------------------------------------

    inputs = tf.stack([logk, b, logz], axis=1)
    policy_out = policy_net(inputs)
    
    # Policy outputs:
    #   dlog(k): change in log-capital
    #   b_next : next-period debt choice

    delta_logk = policy_out[:, 0]
    b_next = B_MAX * tf.tanh(policy_out[:,1])         # Constrain debt to feasible bounds via tanh scaling
    
    # Next-period capital in log form and level
    logk_next = logk + delta_logk
    k_next = tf.exp(logk_next)

    # Investment implied by the policy: I = k' − (1-DELTA) k
    I = k_next - (1.0 - DELTA) * k

    # --------------------------------------------------------
    # Two independent productivity shocks (AiO)
    # --------------------------------------------------------

    logz1 = RHO_Z * logz + SIGMA_EPS * eps1
    logz2 = RHO_Z * logz + SIGMA_EPS * eps2

    # --------------------------------------------------------
    # Continuation value tomorrow
    # --------------------------------------------------------
    
    # V(k', b', z1')
    V1 = tf.squeeze(value_net(tf.stack([logk_next, b_next, logz1], axis=1)), axis=1)
    # V(k', b', z2')
    V2 = tf.squeeze(value_net(tf.stack([logk_next, b_next, logz2], axis=1)), axis=1)

    # --------------------------------------------------------
    # Default indicators
    # --------------------------------------------------------
    
    # Default occurs if continuation equity value is non-positive
    default1 = tf.cast(V1 <= 0.0, tf.float32)
    default2 = tf.cast(V2 <= 0.0, tf.float32)

    # --------------------------------------------------------
    # Recovery values upon default
    # --------------------------------------------------------

    R1 = recovery(k_next, tf.exp(logz1))

    # --------------------------------------------------------
    # Risky interest rate
    # --------------------------------------------------------

    # Probability of no default and expected recovery
    P_no_default = tf.reduce_mean(1.0 - default1)
    E_recovery = tf.reduce_mean(default1 * R1)
    
    # Risky interest rate implied by zero-profit condition of lenders
    r_tilde = (
        ((1.0 + RISK_FREE_R) * b_next - E_recovery)/ (b_next * P_no_default + 1e-8) - 1.0
    )

    # --------------------------------------------------------
    # Current-period flow payoff
    # --------------------------------------------------------

    e = (
        (1.0 - TAU) * profit(k, z)
        - adj_cost(I, k)
        - I
        + b_next / (1.0 + r_tilde)
        + (TAU * r_tilde * b_next)/ ((1.0 + r_tilde) * (1.0 + RISK_FREE_R))
        - b
    )
    
    # --------------------------------------------------------
    # Bellman residuals
    # --------------------------------------------------------
    
    # Current value
    V = tf.squeeze(value_net(inputs), axis=1)
    
    # Continuation values
    C1 = e + BETA * V1
    C2 = e + BETA * V2
    
    # Equity value truncated at zero (default)
    V_target1 = tf.maximum(0.0, C1)
    V_target2 = tf.maximum(0.0, C2)
    
    # Normalization for numerical stability
    scale = 1.0 + tf.abs(V)
    RV1 = (V - V_target1) / scale
    RV2 = (V - V_target2) / scale

    # --------------------------------------------------------
    # AiO loss
    # --------------------------------------------------------

    loss = tf.reduce_mean(RV1 * RV2)
    return loss
