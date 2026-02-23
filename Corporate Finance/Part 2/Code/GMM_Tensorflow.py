"""
Generalized Method of Moments (GMM) Estimation
=============================================

The goal of GMM:
---------------
Choose structural parameters = (theta, rho, sigma) such that
the model-implied Euler equation moments are orthogonal to a set of valid instruments.
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
# Risk-free discount rate r
r = tf.constant(0.04, dtype=tf.float64)

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
rho_true = tf.constant(0.9, dtype=tf.float64)
# Standard deviation of productivity shocks
sigma_true = tf.constant(0.1, dtype=tf.float64)

# ======================================================
# DATA SIMULATION FUNCTION 
# ======================================================

def simulate_data_tf(theta, rho, sigma_eps, T, burn_in, seed=42):
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
# EULER EQUATION RESIDUAL
# ======================================================

def euler_residual(k, z, I, theta):
    """
    Computes the Euler equation residual u_t.

    Euler equation:

    1 + psi_I(I_t, k_t) =
        E_t [ (pi_k(k_{t+1}, z_{t+1})
               − psi_k(I_{t+1}, k_{t+1})
               + (1 − delta)(1 + psi_I(I_{t+1}, k_{t+1}))
             ) / (1 + r) ]

    The residual is:
        u_t = LHS − RHS
    """

    # psi_I = psi_0 (I_t / k_t)
    def psi_I(I, k):
        return psi_0 * (I / k)

    # psi_k = − (psi_0 / 2) (I / k)^2
    def psi_k(I, k):
        return -0.5 * psi_0 * tf.square(I / k)

    # pi_k = theta z k^{theta−1}
    def pi_k(k, z):
        return theta * z * tf.pow(k, theta - 1.0)

    # Compute components
    psi_I_t = psi_I(I[:-1], k[:-1])
    psi_I_tp1 = psi_I(I[1:], k[1:])
    psi_k_tp1 = psi_k(I[1:], k[1:])
    pi_k_tp1 = pi_k(k[1:], z[1:])

    # RHS of Euler equation
    rhs = (
        pi_k_tp1
        - psi_k_tp1
        + (1 - delta) * (1 + psi_I_tp1)
    ) / (1 + r)

    # Euler residual
    return (1.0 + psi_I_t) - rhs


# ======================================================
# GMM MOMENT CONDITIONS
# ======================================================

def get_combined_moments(k, z, I, theta):
    """
    Constructs the vector of GMM moment conditions.

    Moment vector:
    --------------
    m1 = E[u_t * z_t]      (Euler × instrument)
    m2 = E[u_t * k_t]      (Euler × instrument)
    m3 = E[u_t]            (Euler equation itself)
    m4 = Corr(log z_t, log z_{t−1})   (AR(1) persistence)
    m5 = Var(log z_t)                  (shock variance)
    """

    def corr(a, b):
        """
        Computes Pearson correlation coefficient:
        Corr(a, b) = Cov(a, b) / (sigma_a sigma_b)
        """
        return tf.reduce_mean(
            (a - tf.reduce_mean(a)) * (b - tf.reduce_mean(b))
        ) / (tf.math.reduce_std(a) * tf.math.reduce_std(b))

    # Euler residual
    u = euler_residual(k, z, I, theta)

    # Instruments 
    z_t = z[:-1]
    k_t = k[:-1]

    # Euler-based moments
    m1 = tf.reduce_mean(u * z_t)
    m2 = tf.reduce_mean(u * k_t)
    m3 = tf.reduce_mean(u)

    # Productivity process moments
    logz = tf.math.log(z)
    m4 = corr(logz[1:], logz[:-1])
    m5 = tf.math.reduce_variance(logz)

    return tf.stack([m1, m2, m3, m4, m5])


# ======================================================
# PARAMETERS TO BE ESTIMATED
# ======================================================

theta_v = tf.Variable(0.5, dtype=tf.float64)
rho_v = tf.Variable(0.8, dtype=tf.float64)
sigma_v = tf.Variable(0.08, dtype=tf.float64)

# ======================================================
# GMM OBJECTIVE FUNCTION
# ======================================================

# Weighting matrix 
W = tf.eye(5, dtype=tf.float64)

# Generate "real" data
k_real, z_real, I_real = simulate_data_tf(theta_true, rho_true, sigma_true, T, burn_in)
# Data moments 
data_moments = get_combined_moments(k_real, z_real, I_real, theta_true)

def gmm_loss(theta, rho, sigma, W):
    """
    GMM objective: Q(params) = g(params)' W g(params)

    where:
        g(params) = model_moments − data_moments
    """

    # Simulate model data
    k_sim, z_sim, I_sim = simulate_data_tf(theta, rho, sigma, T, burn_in, seed=1)

    # Compute model moments
    model_moments = get_combined_moments(k_sim, z_sim, I_sim, theta) 
    diff = model_moments - data_moments

    # loss
    loss = tf.tensordot(tf.tensordot(diff, W, axes=1), diff, axes=1)

    return loss, model_moments, data_moments


# ======================================================
# OPTIMIZATION LOOP
# ======================================================

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_history = []

print("Training GMM estimator...")

for i in range(500):
    with tf.GradientTape() as tape:
        loss, model_moments, data_moments = gmm_loss(theta_v, rho_v, sigma_v, W)
        
    # Gradient-based update
    grads = tape.gradient(loss, [theta_v, rho_v, sigma_v])
    optimizer.apply_gradients(zip(grads, [theta_v, rho_v, sigma_v]))

    loss_history.append(loss.numpy())

    if i % 50 == 0:
        print(f"Iter {i:3} | Loss: {loss.numpy():.6e}")


# ======================================================
# LOSS DIAGNOSTICS
# ======================================================

plt.figure(figsize=(8, 5))
plt.plot(loss_history, lw=2)
plt.title("GMM Optimization: Loss Minimization")
plt.xlabel("Iteration")
plt.ylabel("Loss Q")
plt.yscale("log")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()


# ======================================================
# OVERIDENTIFICATION TEST (J-TEST)
# ======================================================

N = T - burn_in
J_stat = N * loss_history[-1]
p_value = 1 - chi2.cdf(J_stat, df=5 - 3)

# ======================================================
# RESULTS TABLES
# ======================================================

print("\n" + "=" * 50)
print(f"{'GMM ESTIMATION RESULTS':^50}")
print("=" * 50)

# ======================================================
# TABLE 1: PARAMETER ESTIMATES
# ======================================================
print(f"\n{'TABLE 1: PARAMETER ESTIMATES':^50}")
print("-" * 50)
print(f"{'Parameter':<15} | {'True':<10} | {'Estimated':<12}")
print("-" * 50)
print(f"Theta           | {theta_true.numpy():<10.4f} | {theta_v.numpy():<12.4f}")
print(f"Rho             | {rho_true.numpy():<10.4f} | {rho_v.numpy():<12.4f}")
print(f"Sigma           | {sigma_true.numpy():<10.4f} | {sigma_v.numpy():<12.4f}")
print("-" * 50)

# ======================================================
# TABLE 2: MOMENTS COMPARISON 
# ======================================================

print(f"\n{'TABLE 2: MOMENTS COMPARISON':^50}")
print("-" * 50)
print(f"{'Moment Description':<30} | {'Data':<8} | {'Model':<8}")
print("-" * 50)

moment_names = [
    "Mean Euler Residual",
    "Euler × Productivity",
    "Euler × Capital",
    "Corr(log z_t, log z_{t-1})",
    "Var(log z)"
]

for i, name in enumerate(moment_names):
    print(f"{name:<30} | "f"{data_moments[i].numpy():<8.4f} | "f"{model_moments[i].numpy():<8.4f}")
print("-" * 50)

# ======================================================
# MODEL FIT STATISTICS
# ======================================================
print(f"\n{'MODEL FIT STATISTICS':^50}")
print("-" * 50)
print(f"Loss Q      : {loss_history[-1]:.6f}")
print(f"J-statistic    : {J_stat:.4f}")
print(f"P-value        : {p_value:.4f}")
print("=" * 50)