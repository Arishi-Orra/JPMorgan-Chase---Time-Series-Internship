import tensorflow as tf

###############################################################################
# PolicyNet
# ----------
# A neural network that represents the firm's optimal investment policy:
#
#     k' = h_theta(k, z)
#
# Inputs:
#   - inputs: Tensor of shape (batch, 2), representing (log(k), log(z))
#
# Output:
#   - A single scalar per sample: dlog(k), i.e., the change in log-capital:
#
#         log(k_next) = log(k) + dlog(k)
#         k_next = exp(log(k_next))
# ----------
###############################################################################

class PolicyNet(tf.keras.Model):
    def __init__(self, hidden_sizes=(64,64)):
        super().__init__()
        # Create hidden layers with each Dense layer followed by tanh
        self.hidden = [tf.keras.layers.Dense(h, activation='tanh') for h in hidden_sizes]
       # Output layer: predicts Delta log(k), i.e., change in log-capital.
       # No activation --> unbounded real output (can increase or decrease capital).
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        # Output Delta log(k)
        return self.out(x)


###############################################################################
# ValueNet
# --------
# Neural network representing the value function:
#
#     V(k, z)
#
# Inputs:
#   - inputs: Tensor of shape (batch, 2) = (log(k), log(z))
#
# Outputs:
#   - A scalar V for each sample (unbounded real number)
# -----------------
###############################################################################

class ValueNet(tf.keras.Model):
    def __init__(self, hidden_sizes=(64,64)):
        super().__init__()
        self.hidden = [tf.keras.layers.Dense(h, activation='tanh') for h in hidden_sizes]
        # Output: scalar V(k,z)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

