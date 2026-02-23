"""
Bayesian Estimation using Hamiltonian Monte Carlo (HMC)
=======================================================

The goal of Bayesian estimation:
--------------------------------
Estimate the structural parameters by sampling from the posterior distribution.
"""

import tensorflow as tf
import tensorflow_probability as tfp

# tfd: probabilistic distributions (Beta, Normal) used for priors and likelihoods
tfd = tfp.distributions

# tfb: bijectors for parameter transformations (Sigmoid)
tfb = tfp.bijectors
dtype = tf.float64

# ======================================================
# FIXED (CALIBRATED) PARAMETERS
# ======================================================

# Capital depreciation rate 
delta = tf.constant(0.10, dtype=tf.float64)
# Quadratic investment adjustment cost parameter 
psi_0 = tf.constant(0.01, dtype=tf.float64)
# Standard deviation of measurement error; 
# used to smooth the likelihood surface in the Particle Filter
sigma_me = tf.constant(0.01, dtype=dtype)

# ======================================================
# TRUE PARAMETERS
# ======================================================

# Production function curvature parameter
theta_true = tf.constant(0.6, dtype=tf.float64)
# Persistence of productivity shocks (AR(1) coefficient)
rho_true = tf.constant(0.8, dtype=tf.float64)
# Standard deviation of productivity shocks
sigma_true = tf.constant(0.2, dtype=tf.float64)
true_vals = [theta_true, rho_true, sigma_true]

# ======================================================
# DATA SIMULATION
# ======================================================
def simulate_observed_data(theta, rho, sigma, T=200, burn_in=100):
    """
    Simulates capital, productivity, and investment paths 
    from the dynamic investment model.

    Parameters
    ----------
    theta : production curvature parameter 
    rho : AR(1) persistence of productivity
    sigma : standard deviation of productivity shocks
    T : total simulation length
    burn_in : number of initial observations dropped

    Returns
    -------
    I : investment series after burn-in
    """
    
    tf.random.set_seed(999)
    
    # Initialization
    k_prev = tf.constant(1.0, dtype=dtype)
    z_prev = tf.constant(1.0, dtype=dtype)
    I_list = []
    # Productivity shocks eps_t ~ N(0, sigma_eps^2)
    eps = tf.random.normal([T + burn_in], 0.0, sigma, dtype=dtype)
    I_prev_val = delta

    for t in range(T + burn_in):
        # Productivity process: log z_t = rho log z_{t-1} + eps_t
        z_t = tf.exp(rho * tf.math.log(z_prev) + eps[t])
        
        # Investment adjustment cost term
        chi = 1.0 + psi_0 * (I_prev_val / k_prev)
        
        # Optimal target capital (static FOC): theta z_t k^{theta-1} = X_t
        # => k* = (theta z_t / X_t)^{1/(1-theta)}
        k_target = tf.pow((theta * z_t) / chi, 1.0 / (1.0 - theta))
        
        # k_t = (1 - delta) k_{t-1} + delta k*
        k_t = (1.0 - delta) * k_prev + delta * k_target
        
        # I_t = k_t - (1 - delta) k_{t-1}
        i_t = k_t - (1.0 - delta) * k_prev
        
        # Collect data only after the burn-in period to remove initial condition bias
        if t >= burn_in:
            I_list.append(i_t)
            
        # Update state variables for the next period's iteration
        k_prev, z_prev, I_prev_val = k_t, z_t, i_t
    return tf.stack(I_list)


# ======================================================
# PARTICLE FILTER LIKELIHOOD
# ======================================================
def particle_filter_likelihood(obs_data, theta, rho, sigma, n_particles=50):
    """
    Computes the log-likelihood of the model parameters using a Particle Filter.

    Parameters
    ----------
    obs_data : Tensor of observed investment values
    theta : production curvature parameter
    rho : AR(1) persistence of productivity
    sigma : standard deviation of productivity shocks
    n_particles : number of particles used to approximate the distribution

    Returns
    -------
    total_log_lik : the accumulated log-likelihood of the observed data
    """
    # number of time periods
    T_len = obs_data.shape[0]
    # Define a constant for 1.0
    one = tf.constant(1.0, dtype=dtype)
    
    # Stationary variance of AR(1):
    z_init_std = sigma / tf.sqrt(one - tf.square(rho))
    
    # Initialize particles
    z_p = tf.exp(tf.random.normal([n_particles], 0.0, z_init_std, dtype=dtype))
    k_p = tf.ones([n_particles], dtype=dtype)
    I_prev_p = tf.fill([n_particles], delta)
    
    total_log_lik = tf.constant(0.0, dtype=dtype)

    for t in range(T_len):
        # State transition
        eps_p = tf.random.normal([n_particles], 0.0, sigma, dtype=dtype)
        z_p = tf.exp(rho * tf.math.log(tf.maximum(z_p, 1e-10)) + eps_p)
        
        # Model-implied investment
        chi = one + psi_0 * (I_prev_p / k_p)
        k_target = tf.pow((theta * z_p) / chi, one / (one - theta))
        k_next = (one - delta) * k_p + delta * k_target
        I_pred = k_next - (one - delta) * k_p
        
        # Compute the log-probability of the observed data point
        log_weights = tfd.Normal(loc=I_pred, scale=sigma_me).log_prob(obs_data[t])
        
        # Identify the maximum log-weight to facilitate a stable Log-Sum-Exp calculation
        max_lw = tf.reduce_max(log_weights)
        
        # Shift the log-weights by the maximum to prevent numerical overflow when exponentiating
        relative_weights = tf.exp(log_weights - max_lw)
        
        # Accumulate Log-Likelihood 
        total_log_lik += max_lw + tf.math.log(tf.reduce_mean(relative_weights))
        
        # Normalize the relative weights to create a valid probability distribution for resampling
        resampling_weights = relative_weights / tf.reduce_sum(relative_weights)
        
        # Sample new particle indices based on their weights
        idx = tf.stop_gradient(tf.random.categorical(tf.math.log([resampling_weights]), n_particles)[0])
        
        # Resampling
        z_p = tf.gather(z_p, idx)
        k_p = tf.gather(k_next, idx)
        I_prev_p = tf.gather(I_pred, idx)
        
    return total_log_lik



# ======================================================
# HMC ESTIMATION
# ======================================================
def run_bayesian_estimation(data):
    """
    Samples structural parameters from the posterior distribution using HMC.

    Parameters
    ----------
    data : Tensor of observed investment series used for estimation

    Returns
    -------
    post_samples : List containing posterior samples for theta, rho, and sigma
    """
    
    # Define a Sigmoid bijector to map unconstrained real numbers to the [0.2, 0.9] range for theta
    theta_bij = tfb.Sigmoid(low=tf.constant(0.2, dtype=dtype), high=tf.constant(0.9, dtype=dtype))
    
    # Define a Sigmoid bijector to map unconstrained real numbers to the [0.7, 0.99] range for rho
    rho_bij   = tfb.Sigmoid(low=tf.constant(0.7, dtype=dtype), high=tf.constant(0.99, dtype=dtype))
    
    # Define a Softplus bijector to ensure the volatility parameter sigma remains strictly positive
    sigma_bij = tfb.Softplus()

    # Define the target log-probability function that HMC will explore
    def target_log_prob(u_theta, u_rho, u_sigma):
        # Transform unconstrained variables into valid parameters using the bijectors
        theta = theta_bij.forward(u_theta)
        rho   = rho_bij.forward(u_rho)
        sigma = sigma_bij.forward(u_sigma)
        
        # Priors
        lp = (tfd.Beta(tf.constant(2.0, dtype=dtype), tf.constant(2.0, dtype=dtype)).log_prob((theta-0.2)/0.7) +
              tfd.Beta(tf.constant(2.0, dtype=dtype), tf.constant(2.0, dtype=dtype)).log_prob((rho-0.7)/0.29) +
              tfd.HalfNormal(scale=tf.constant(0.1, dtype=dtype)).log_prob(sigma))
        
        # Likelihood using particle filter
        ll = particle_filter_likelihood(data, theta, rho, sigma)

        # Jacobian adjustment:
        # Since HMC samples in an unconstrained space while our parameters are bounded, 
        # we must add the log-determinant of the Jacobian of the transformation. 
        # This corrects for the stretching of the probability density caused by the bijectors
        jacobian = (theta_bij.forward_log_det_jacobian(u_theta, event_ndims=0) +
                    rho_bij.forward_log_det_jacobian(u_rho, event_ndims=0) +
                    sigma_bij.forward_log_det_jacobian(u_sigma, event_ndims=0))
        
        return ll + lp + jacobian

    # Initialize the HMC algorithm with the leapfrog integrator and target probability function
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        num_leapfrog_steps=5,
        step_size=0.01
    )

    # Wrap the HMC kernel with an adaptive step-size optimizer for better chain convergence
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel, num_adaptation_steps=80
    )

    # Set initial parameter guesses
    init_state = [theta_bij.inverse(tf.constant(0.5, dtype=dtype)), 
                  rho_bij.inverse(tf.constant(0.9, dtype=dtype)), 
                  sigma_bij.inverse(tf.constant(0.08, dtype=dtype))]

    # Print status and begin the MCMC sampling process
    print("Running HMC...")
    samples = tfp.mcmc.sample_chain(
        num_results=300,
        num_burnin_steps=100,
        current_state=init_state,
        kernel=adaptive_kernel,
        trace_fn=None 
    )
    
    # Transform the final posterior samples to constrained space
    return [theta_bij.forward(samples[0]), 
            rho_bij.forward(samples[1]), 
            sigma_bij.forward(samples[2])]

# ======================================================
# EXECUTION & DIAGNOSTICS
# ======================================================

# Generate Synthetic Data from known truths
obs_I = simulate_observed_data(theta_true, rho_true, sigma_true)

# Execute the Bayesian estimation framework using the simulated data
theta_post, rho_post, sigma_post = run_bayesian_estimation(obs_I)

# posterior samples
post_samples = [theta_post, rho_post, sigma_post]
names = ['Theta', 'Rho', 'Sigma']

print("\n--- Parameter Recovery Results ---")
for name, post, true in zip(names, post_samples, true_vals):
    # Calculate the mean of the posterior samples
    mean_val = tf.reduce_mean(post)
    # Print the comparison between the 'Known Truth' and our 'Estimated Mean'
    print(f"{name}: True={true.numpy():.4f}, Estimated Mean={mean_val:.4f}")