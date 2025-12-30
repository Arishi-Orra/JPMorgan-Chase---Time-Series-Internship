"""
The training loop:
    1) Samples states and shocks
    2) Computes the AiO loss
    3) Performs gradient updates on both networks
    4) Periodically evaluates the lifetime value and model loss
"""

import tensorflow as tf

from src.networks import PolicyNet, ValueNet
from src.losses import bellman_aio_loss
from src.utils import sample_batch
from src.testing import compute_lifetime_reward

def train_model(num_steps=100000, batch_size=512, lambda_foc=1.0, lr=1e-3, eval_every=500):
    
    # Initialize networks
    policy_net = PolicyNet()            # maps (logk, logz) --> dlogk
    value_net  = ValueNet()             # maps (logk, logz) --> V(k,z)

    optim = tf.keras.optimizers.Adam(lr)         # Optimizer shared by both networks
    
    # Training logs
    loss_history = []
    lr_history = []             # lifetime reward evaluation history
    eval_steps = []             # steps at which evaluation occurred

    for step in range(1, num_steps+1):
        # ---- Sample random batch of states and shocks ----
        logk, logz, eps1, eps2 = sample_batch(batch_size)
        # ---- Compute AiO loss and its gradients ----
        with tf.GradientTape() as tape:
            loss = bellman_aio_loss(policy_net, value_net, logk, logz, eps1, eps2, lambda_foc)

        # Joint update of policy and value network parameters
        grads = tape.gradient(loss, policy_net.trainable_variables + value_net.trainable_variables)
        optim.apply_gradients(zip(grads, policy_net.trainable_variables + value_net.trainable_variables))

        loss_history.append(float(loss))
        
        # Periodic evaluation: compute lifetime value
        # if step % eval_every == 0:
        mean_LR, _, _ = compute_lifetime_reward(policy_net, value_net)
        lr_history.append(mean_LR)
        eval_steps.append(step)
        print(f"[Step {step}/{num_steps}] Loss={loss:.5f}, LR={mean_LR:.3f}")
            
    # Return trained models and logs
    return policy_net, value_net, loss_history, lr_history, eval_steps
