"""
xor_test.py
Train and recall a 2-3-1 MLP on the Boolean XOR function (Task 3.4).

Architecture
    2 inputs, 3 hidden units, 1 output
    η = 0.7, momentum = 0.9, 10 000 epochs of online BP

XOR is not linearly separable; the hidden layer builds the intermediate
features (approximately OR and NAND, or the two diagonals) that the
output unit can then combine linearly.
"""

import random
from network import Network

random.seed(7)

xor_input = [[0, 0], [1, 0], [0, 1], [1, 1]]
xor_ideal = [[0], [1], [1], [0]]

# 2 inputs → 3 hidden → 1 output
network = Network(2, 3, 1, 0.7, 0.9)

for epoch in range(10000):
    for x, y in zip(xor_input, xor_ideal):
        network.compute_outputs(x)
        network.calc_error(y)
        network.learn()
    rmse = network.get_error(len(xor_input))
    if (epoch + 1) % 1000 == 0 or epoch == 0:
        print("epoch %5d  RMSE = %.6f" % (epoch + 1, rmse))

print("Recall")
for x in xor_input:
    print(x, "->", round(network.compute_outputs(x)[0], 4))
