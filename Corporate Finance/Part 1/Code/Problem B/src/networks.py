import tensorflow as tf

###############################################################################
# PolicyNet
# ----------
# Neural network representing the firm's optimal joint policy for
# investment and debt issuance in the risky-debt model:
#
#     (k', b') = h_theta(k, b, z)
#
# Inputs:
#   - inputs: Tensor of shape (batch, 3), representing:
#
#         ( log(k), b, log(z) )
#
# Outputs:
#   - Two scalars per sample: [ dlog(k), b_next ]
#
#       log(k_next) = log(k) + dlog(k)
#       k_next      = exp(log(k_next))
#       b_next      = next-period debt choice (level)
# ----------
###############################################################################

class PolicyNet(tf.keras.Model):
    def __init__(self, hidden_sizes=(64,64)):
        super().__init__()

        # Hidden layers with tanh activations
        self.hidden = [
            tf.keras.layers.Dense(h, activation='tanh')
            for h in hidden_sizes
        ]

        # Output layer:
        #   dlog(k): change in log-capital
        #   b_next : next-period debt level
        self.out = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)


###############################################################################
# ValueNet
# --------
# Neural network representing the value function:
#
#     V(k, b, z)
#
# Inputs:
#   - inputs: Tensor of shape (batch, 3), representing:
#
#         ( log(k), b, log(z) )
#
# Outputs:
#   - A single scalar per sample:
#
#         V(k, b, z)
# --------
###############################################################################


class ValueNet(tf.keras.Model):
    def __init__(self, hidden_sizes=(64,64)):
        super().__init__()
        # Hidden layers with tanh activations
        self.hidden = [
            tf.keras.layers.Dense(h, activation='tanh')
            for h in hidden_sizes
        ]
        # Output layer: scalar value V(k,b,z)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)