import tensorflow as tf
from src.config import ALPHA, PHI_ADJ

# ----------------------------------------
# Helper functions: pi, psi, psi_I
# ----------------------------------------

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

def adj_cost_I(I, k):
    """
    Derivative of psi(I,k) wrt I:
      psi_I = PHI_ADJ * I / k
    """
    return PHI_ADJ * I / (k + 1e-12)