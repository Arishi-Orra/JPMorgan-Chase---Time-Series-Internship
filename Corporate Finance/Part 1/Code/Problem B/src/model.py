import tensorflow as tf
from src.config import ALPHA, PHI_ADJ, TAU, ALPHA_BK, DELTA

# --------------------------------------------
# Helper functions: pi, psi, Recovery value
# --------------------------------------------

def profit(k, z):
    """
    Profit function: pi(k,z) = z * k^ALPHA
    """
    return z * tf.pow(k, ALPHA)


def adj_cost(I, k):
    """
    Quadratic adjustment cost:
      psi(I,k) = 0.5 * PHI_ADJ * I^2 / k
    """
    return 0.5 * PHI_ADJ * tf.square(I) / (k + 1e-12)


def recovery(k_next, z_next):
    """
    Recovery value for lenders in default:

        R = (1 - ALPHA_BK) * [ (1 - TAU) * pi(k',z') + (1 - DELTA) * k' ]
    """
    return (1.0 - ALPHA_BK) * (
        (1.0 - TAU) * profit(k_next, z_next) + (1.0 - DELTA) * k_next
    )