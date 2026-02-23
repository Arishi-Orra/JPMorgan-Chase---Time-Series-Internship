"""
Simulated Method of Moments (SMM) Estimation
==========================================

The goal of SMM:
----------------
Choose structural parameters = (theta, row, sigma) such that
simulated moments match moments computed from "real" data.

"""

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import chi2


# ======================================================
# FIXED (CALIBRATED) PARAMETERS
# ======================================================

# Capital depreciation rate 
delta = tf.constant(0.10, dtype=tf.float64)
# Quadratic investment adjustment cost parameter 
psi_0 = tf.constant(0.01, dtype=tf.float64)

# Total simulation length
T = 1500
# Burn-in period to eliminate dependence on initial conditions
burn_in = 200


# ======================================================
# TRUE PARAMETERS
# ======================================================

# Production function curvature parameter
theta_true = tf.constant(0.6, dtype=tf.float64)
# Persistence of productivity shocks (AR(1) coefficient)
rho_true = tf.constant(0.8, dtype=tf.float64)
# Standard deviation of productivity shocks
sigma_eps_true = tf.constant(0.2, dtype=tf.float64)


# ======================================================
# DATA SIMULATION FUNCTION 
# ======================================================

def simulate_data_tf(theta, rho, sigma_eps, T, burn_in, seed=999):
    """
    Simulates capital, productivity, and investment paths
    from the dynamic investment model.

    Parameters
    ----------
    theta : production curvature parameter 
    rho : AR(1) persistence of productivity
    sigma_eps : standard deviation of productivity shocks
    T : total simulation length
    burn_in : number of initial observations dropped
    seed : random seed for reproducibility

    Returns
    -------
    k : capital series after burn-in
    z : productivity series after burn-in
    I : investment series after burn-in
    """

    tf.random.set_seed(seed)

    # Lists for storing simulated paths
    k_list, z_list, I_list = [], [], []

    # Initial conditions
    k_prev = tf.constant(1.0, dtype=tf.float64)
    z_prev = tf.constant(1.0, dtype=tf.float64)

    # Productivity shocks eps_t ~ N(0, sigma_eps^2)
    eps = tf.random.normal([T], mean=0.0, stddev=sigma_eps, seed=seed, dtype=tf.float64)

    for t in range(1, T):
        # Productivity process: log z_t = rho log z_{t-1} + eps_t
        z_t = tf.exp(rho * tf.math.log(z_prev) + eps[t])

        # Investment adjustment cost term: X_t = 1 + psi_0 * (I_{t-1} / k_{t-1})
        I_prev_rate = (I_list[-1] / k_list[-1]) if t > 1 else delta
        chi = 1.0 + psi_0 * I_prev_rate

        # Optimal target capital (static FOC): theta z_t k^{theta-1} = X_t
        # => k* = (theta z_t / X_t)^{1/(1-theta)}
        k_target = tf.pow(theta * z_t / chi, 1.0 / (1.0 - theta))

        # k_t = (1 - delta) k_{t-1} + delta k*
        k_t = (1.0 - delta) * k_prev + delta * k_target

        # I_t = k_t - (1 - delta) k_{t-1}
        i_t = k_t - (1.0 - delta) * k_prev

        # Store values
        k_list.append(k_t)
        z_list.append(z_t)
        I_list.append(i_t)

        # Update states
        k_prev = k_t
        z_prev = z_t

    # Drop burn-in observations
    k = tf.stack(k_list)[burn_in:]
    z = tf.stack(z_list)[burn_in:]
    I = tf.stack(I_list)[burn_in:]

    return k, z, I


# ======================================================
# MOMENT COMPUTATION
# ======================================================

def get_moments(k, z, I):
    """
    Computes model-implied moments used for SMM estimation.

    Moments:
    --------
    m1 = E[I_t / k_t]
    m2 = Corr(I_t / k_t, z_t)
    m3 = Corr(log z_t, log z_{t-1})
    m4 = Var(log z_t)
    """

    inv_rate = I / k
    lnz = tf.math.log(z)

    def corr(a, b):
        """
        Computes Pearson correlation coefficient:
        Corr(a, b) = Cov(a, b) / (sigma_a sigma_b)
        """
        return tf.reduce_mean(
            (a - tf.reduce_mean(a)) * (b - tf.reduce_mean(b))
        ) / (tf.math.reduce_std(a) * tf.math.reduce_std(b))

    # Mean investment rate
    m1 = tf.reduce_mean(inv_rate)
    # Correlation between investment rate and productivity
    m2 = corr(inv_rate, z)
    # AR(1) persistence of productivity
    m3 = corr(lnz[1:], lnz[:-1])
    # Variance of log productivity
    _, m4 = tf.nn.moments(lnz, axes=[0])

    return tf.stack([m1, m2, m3, m4])


# ======================================================
# GENERATE "REAL" DATA MOMENTS
# ======================================================

k_real, z_real, I_real = simulate_data_tf(theta_true, rho_true, sigma_eps_true, T, burn_in)
real_moments = get_moments(k_real, z_real, I_real)

# ======================================================
# PARAMETERS TO BE ESTIMATED
# ======================================================

theta_v = tf.Variable(0.5, dtype=tf.float64)
rho_v = tf.Variable(0.9, dtype=tf.float64)
sigma_v = tf.Variable(0.08, dtype=tf.float64)

# ======================================================
# SMM OPTIMIZATION
# ======================================================

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_history = []

print("Training SMM...")

for i in range(500):
    with tf.GradientTape() as tape:
        # Simulate data with candidate parameters
        k_sim, z_sim, I_sim = simulate_data_tf(theta_v, rho_v, sigma_v, T, burn_in, seed=15)
        
        # Compute simulated moments
        sim_moments = get_moments(k_sim, z_sim, I_sim)

        # SMM objective: Q = (m_data - m_sim)'(m_data - m_sim)
        smm_loss = tf.reduce_sum(tf.square(real_moments - sim_moments))

    # Gradient-based update
    grads = tape.gradient(smm_loss, [theta_v, rho_v, sigma_v])
    optimizer.apply_gradients(zip(grads, [theta_v, rho_v, sigma_v]))

    loss_history.append(smm_loss.numpy())
    if i % 50 == 0:
        print(f"Iter {i:3} | Loss: {smm_loss.numpy():.6e}")


# ======================================================
# LOSS DIAGNOSTICS
# ======================================================

plt.figure(figsize=(8, 5))
plt.plot(loss_history, lw=2)
plt.title("SMM Optimization: Loss Minimization")
plt.xlabel("Iteration")
plt.ylabel("Loss Q")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# ======================================================
# OVERIDENTIFYING TEST (J-TEST)
# ======================================================

N = T - burn_in
M = 1.0  # Ratio of simulated to real sample size

J_stat = (N * M / (1 + M)) * smm_loss.numpy()
df = 4 - 3  # moments - parameters
p_value = 1 - chi2.cdf(J_stat, df)

# ======================================================
# RESULTS TABLES
# ======================================================

print("\n" + "=" * 50)
print(f"{'SMM ESTIMATION RESULTS':^50}")
print("=" * 50)

# ======================================================
# TABLE 1: PARAMETER ESTIMATES
# ======================================================

print(f"\n{'TABLE 1: PARAMETER ESTIMATES':^50}")
print("-" * 50)
print(f"{'Parameter':<15} | {'True':<10} | {'Estimated':<10}")
print("-" * 50)
print(f"Theta         | {theta_true.numpy():<10.4f} | {theta_v.numpy():<10.4f}")
print(f"Rho           | {rho_true.numpy():<10.4f} | {rho_v.numpy():<10.4f}")
print(f"Sigma         | {sigma_eps_true.numpy():<10.4f} | {sigma_v.numpy():<10.4f}")
print("-" * 50)

# ======================================================
# TABLE 2: MOMENTS COMPARISON 
# ======================================================

print(f"\n{'TABLE 2: MOMENTS COMPARISON':^50}")
print("-" * 50)
print(f"{'Moment Description':<30} | {'Data':<8} | {'Model':<8}")
print("-" * 50)
moment_names = [
    "Mean(I/k)",
    "Corr(I/k, z)",
    "Corr(log z_t, log z_{t-1})",
    "Var(log z)"
]

for i, name in enumerate(moment_names):
    print(f"{name:<30} | {real_moments[i].numpy():<8.4f} | {sim_moments[i].numpy():<8.4f}")
print("-" * 50)

# ======================================================
# MODEL FIT STATISTICS
# ======================================================
print(f"\n{'MODEL FIT STATISTICS':^50}")
print("-" * 50)
print(f"Loss Q      : {smm_loss.numpy():.6f}")
print(f"J-statistic    : {J_stat:.4f}")
print(f"P-value        : {p_value:.4f}")
print("=" * 50)