import numpy as np


# Returns the mean squared error or mean squared error derivative of two numpy arrays.
def mse(real: np.array, pred: np.array, d=False) -> float:
    if not d:
        return np.average(np.power(real-pred, 2))
    else:
        return 2 * (pred - real) / real.size
