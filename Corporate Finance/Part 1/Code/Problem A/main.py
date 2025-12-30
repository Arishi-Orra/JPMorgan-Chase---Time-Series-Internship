"""
1) Sets a reproducible seed
2) Trains the policy and value networks using the Bellman-AiO loss
3) Plots training diagnostics (loss curve, lifetime reward curve)
4) Computes and visualizes Bellman residuals on a test set
5) Simulates cumulative rewards under the learned policy
"""

from src.utils import set_seed
from src.training import train_model
from src.testing import compute_bellman_residuals, compute_lifetime_reward
import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# 1. Ensure reproducibility
# =====================================================================
set_seed(45)


# =====================================================================
# 2. Train the model: returns trained networks + logs
# =====================================================================
policy_net, value_net, loss_hist, lr_hist, eval_steps = train_model(10000)


# =====================================================================
# 3 Plot training loss curve
# =====================================================================
plt.figure(figsize=(7,4))
plt.plot(loss_hist)
plt.xlabel("Training step")
plt.ylabel("Bellman-AiO loss")
plt.title("Learning curve: loss over training")
plt.grid(True)
plt.show()


# =====================================================================
# 4. Compute and visualize Bellman residual distribution
# =====================================================================
RB = compute_bellman_residuals(policy_net, value_net)
print("Mean Bellman Residual:", np.mean(RB))

# Histogram of the AiO product
plt.hist(np.log10(np.abs(RB) + 1e-12), bins=50)
plt.title("Bellman Residual Histogram (Test Set)")
plt.xlabel("log10 |Residual|")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# =====================================================================
# 5. Simulate cumulative reward under learned policy
# =====================================================================
mean_LR, LR_list, cum_path = compute_lifetime_reward(policy_net, value_net, 30, 100)
print("\nMean Lifetime Reward =", mean_LR)

plt.plot(cum_path)
plt.title("Cumulative Reward Path (Single Simulation)")
plt.xlabel("Time Step")
plt.ylabel("Discounted Reward")
plt.grid(True)
plt.show()
