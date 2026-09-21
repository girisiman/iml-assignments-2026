"""
and_test.py
Train and recall a 2-1-1 MLP on the Boolean AND function (Task 3.2).

Architecture
    2 inputs, 1 hidden unit, 1 output
    η = 0.7, momentum = 0.9, 1000 epochs of online BP

AND is linearly separable, so a single hidden unit (in fact a single
perceptron) is sufficient. The hidden layer is kept to exercise the
same Network class used later for XOR.
"""

import random
from network import Network

random.seed(7)

# Four vertices of the unit square and the AND target
AND_input = [[0, 0], [1, 0], [0, 1], [1, 1]]
AND_ideal = [[0], [0], [0], [1]]

# 2 inputs → 1 hidden → 1 output
network = Network(2, 1, 1, 0.7, 0.9)

# Online training: forward, error, update, once per pattern, 1000 sweeps
for epoch in range(1000):
    for x, y in zip(AND_input, AND_ideal):
        network.compute_outputs(x)   # forward pass, store activations
        network.calc_error(y)        # back-propagate this pattern
        network.learn()              # take one gradient step
    # Report RMSE once per 200 epochs (uses errors accumulated above)
    # Consume this epoch's accumulated SSE so the next epoch starts clean.
    rmse = network.get_error(len(AND_input))
    if (epoch + 1) % 200 == 0 or epoch == 0:
        print("epoch %4d  RMSE = %.6f" % (epoch + 1, rmse))

print("Recall")
for x in AND_input:
    print(x, "->", round(network.compute_outputs(x)[0], 4))
