"""
network.py
Three-layer multilayer perceptron (input – hidden – output) trained by
online backpropagation with momentum.

The implementation follows the assignment skeleton exactly in structure
and arithmetic. Comments added for Task 3.3 explain the role of each
method and the important statements.
"""

import math
import random


class Network:
    """Fully connected 3-layer MLP with logistic (sigmoid) units.

    Parameters
    ----------
    input_count, hidden_count, output_count : int
        Layer widths. Hidden and output units each have a trainable
        threshold (bias), stored separately from the weight matrix.
    learn_rate : float
        Step size η applied to accumulated weight / threshold gradients.
    momentum : float
        Coefficient α on the previous parameter step (accelerates
        consistent directions, damps oscillation).
    """

    def __init__(self, input_count, hidden_count, output_count,
                 learn_rate=0.7, momentum=0.9):
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.input_count = input_count
        self.hidden_count = hidden_count
        self.output_count = output_count

        # Flat index layout: [inputs | hidden | outputs]
        self.neuron_count = input_count + hidden_count + output_count

        # Flat weight layout: all input→hidden weights, then all hidden→output
        self.weight_count = (input_count * hidden_count) + \
                            (hidden_count * output_count)

        self.global_error = 0                      # sum of squared output errors

        self.fire = [0.0] * self.neuron_count      # activations a_i
        self.matrix = [0.0] * self.weight_count    # weights w
        self.matrix_delta = [0.0] * self.weight_count          # previous Δw (momentum)
        self.acc_matrix_delta = [0.0] * self.weight_count      # accumulated ∂E/∂w
        self.thresholds = [0.0] * self.neuron_count            # bias / threshold θ_i
        self.threshold_delta = [0.0] * self.neuron_count
        self.acc_threshold_delta = [0.0] * self.neuron_count
        self.error = [0.0] * self.neuron_count                 # δ-prefactors / backprop signals
        self.error_delta = [0.0] * self.neuron_count           # δ_i = error_i * σ'(net_i)

        self.reset()                               # draw random initial parameters

    def threshold(self, x):
        """Logistic sigmoid σ(x) = 1 / (1 + e^{-x}).

        Maps a net input to (0, 1). Its derivative is σ(x)(1 − σ(x)),
        which is used in calc_error. Numerically saturates for |x| ≫ 0.
        """
        return 1 / (1 + math.exp(-x))

    def compute_outputs(self, inputs):
        """Forward pass.

        Copies the pattern into the input slots of `fire`, then computes
        hidden and output activations:

            net_i = θ_i + Σ_j a_j w_{ji}
            a_i   = σ(net_i)

        Returns the list of output-layer activations.
        """
        hidden_index = self.input_count
        output_index = self.input_count + self.hidden_count

        # Clamp raw inputs into the activation buffer (no sigmoid on inputs)
        for i in range(self.input_count):
            self.fire[i] = inputs[i]

        idx = 0   # running index into the flattened weight matrix

        # Hidden layer: each hidden unit sees every input
        for i in range(hidden_index, output_index):
            s = self.thresholds[i]
            for j in range(self.input_count):
                s += self.fire[j] * self.matrix[idx]
                idx += 1
            self.fire[i] = self.threshold(s)

        result = []

        # Output layer: each output unit sees every hidden unit
        for i in range(output_index, self.neuron_count):
            s = self.thresholds[i]
            for j in range(hidden_index, output_index):
                s += self.fire[j] * self.matrix[idx]
                idx += 1
            self.fire[i] = self.threshold(s)
            result.append(self.fire[i])

        return result

    def calc_error(self, ideal):
        """Back-propagation of one supervised pattern.

        1. Output δ: (t − y) · y · (1 − y)     [chain rule through sigmoid]
        2. Accumulate hidden→output gradients and back-propagate to hidden.
        3. Hidden δ: (Σ_k w_ki δ_k) · h · (1 − h)
        4. Accumulate input→hidden gradients.

        Gradients are *accumulated* in acc_* so that learn() can apply
        them (online, because the assignment calls learn() after each
        pattern). Squared output error is added into global_error.
        """
        hidden_index = self.input_count
        output_index = self.input_count + self.hidden_count

        # Clear hidden/output error accumulators for this pattern
        for i in range(self.input_count, self.neuron_count):
            self.error[i] = 0

        # Output errors: target minus prediction, plus δ = e · σ'
        for i in range(output_index, self.neuron_count):
            self.error[i] = ideal[i - output_index] - self.fire[i]
            self.global_error += self.error[i] ** 2
            self.error_delta[i] = self.error[i] * \
                                  self.fire[i] * (1 - self.fire[i])

        # Start of the hidden→output block in the flat weight vector
        winx = self.input_count * self.hidden_count

        # Hidden-layer error: push output δ back through hidden→output weights
        for i in range(output_index, self.neuron_count):
            for j in range(hidden_index, output_index):
                # ∂E/∂w_{j→i} += δ_i * a_j
                self.acc_matrix_delta[winx] += \
                    self.error_delta[i] * self.fire[j]
                # accumulate incoming error at hidden unit j
                self.error[j] += self.matrix[winx] * self.error_delta[i]
                winx += 1
            self.acc_threshold_delta[i] += self.error_delta[i]

        # Hidden δ = (back-propagated error) · σ'(net)
        for i in range(hidden_index, output_index):
            self.error_delta[i] = self.error[i] * \
                                  self.fire[i] * (1 - self.fire[i])

        # Input→hidden gradients
        winx = 0
        for i in range(hidden_index, output_index):
            for j in range(hidden_index):          # j runs over input units
                self.acc_matrix_delta[winx] += \
                    self.error_delta[i] * self.fire[j]
                self.error[j] += self.matrix[winx] * self.error_delta[i]
                winx += 1
            self.acc_threshold_delta[i] += self.error_delta[i]

    def learn(self):
        """Apply one parameter step using accumulated gradients + momentum.

            Δp ← η · acc_grad + α · Δp_prev
            p  ← p + Δp
            acc_grad ← 0

        Called after every training pattern in the supplied test scripts
        (online / stochastic gradient descent).
        """
        for i in range(len(self.matrix)):
            self.matrix_delta[i] = self.learn_rate * \
                                   self.acc_matrix_delta[i] + \
                                   self.momentum * self.matrix_delta[i]
            self.matrix[i] += self.matrix_delta[i]
            self.acc_matrix_delta[i] = 0

        # Thresholds of hidden and output units only (inputs have none)
        for i in range(self.input_count, self.neuron_count):
            self.threshold_delta[i] = self.learn_rate * \
                                      self.acc_threshold_delta[i] + \
                                      self.momentum * self.threshold_delta[i]
            self.thresholds[i] += self.threshold_delta[i]
            self.acc_threshold_delta[i] = 0

    def get_error(self, length):
        """Root-mean-square error over the last `length` patterns.

            RMSE = sqrt( Σ e² / (N · n_out) )

        Resets global_error so the next call starts a fresh window.
        """
        err = math.sqrt(self.global_error /
                        (length * self.output_count))
        self.global_error = 0
        return err

    def reset(self):
        """Draw small random weights and thresholds in (−0.5, 0.5).

        Symmetric randomisation around zero avoids saturating the
        sigmoid at the start of training. Momentum buffers are zeroed.
        """
        for i in range(self.neuron_count):
            self.thresholds[i] = 0.5 - random.random()
            self.threshold_delta[i] = 0
            self.acc_threshold_delta[i] = 0

        for i in range(len(self.matrix)):
            self.matrix[i] = 0.5 - random.random()
            self.matrix_delta[i] = 0
            self.acc_matrix_delta[i] = 0
