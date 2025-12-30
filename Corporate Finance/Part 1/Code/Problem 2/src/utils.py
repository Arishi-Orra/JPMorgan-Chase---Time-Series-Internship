import tensorflow as tf
import random
import numpy as np

from src.config import (LOGK_MIN, LOGK_MAX, SIGMA_LOGZ, B_MIN, B_MAX)


# ----------------------------------------
# Utility helpers: random seeds, batching
# ----------------------------------------


def set_seed(seed):
    random.seed(seed)                # seed for Python's internal RNG
    np.random.seed(seed)             # seed for NumPy's RNG
    tf.random.set_seed(seed)         # seed for TensorFlow RNG


def sample_batch(batch_size):
    """
    Samples (logk, b, logz) and two independent shocks for AiO expectation.
    """
    
    # Draw log-capital uniformly from [LOGK_MIN, LOGK_MAX]
    logk = tf.random.uniform((batch_size,), LOGK_MIN, LOGK_MAX, dtype=tf.float32)
    
    # Draw log productivity shock from stationary normal distribution
    logz = tf.random.normal((batch_size,), 0.0, SIGMA_LOGZ, dtype=tf.float32)
    
    # Draw outstanding debt level b uniformly from feasible debt bounds
    b = tf.random.uniform((batch_size,), B_MIN, B_MAX, dtype=tf.float32)    

    # Two independent shock realizations for AiO expectation approximation
    eps1 = tf.random.normal((batch_size,), dtype=tf.float32)
    eps2 = tf.random.normal((batch_size,), dtype=tf.float32)

    return logk, b, logz, eps1, eps2