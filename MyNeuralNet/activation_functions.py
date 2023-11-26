import numpy as np


def sigmoid(input, d=False):
    sig = 1 / (1 + np.exp(-input))
    if not d:
        return sig
    if d:
        return sig * (1 - sig)


def relu(input, d=False):
    if not d:
        input[input <= 0] = 0
        return input
    if d:
        input[input > 0] = 1
        input[input < 0] = 0
        return input
