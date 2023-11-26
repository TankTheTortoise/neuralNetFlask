import numpy as np


def split(array: np.array, train_size: float) -> tuple[np.array, np.array]:
    array_train = array[:int(len(array) * train_size)]
    array_test = array[int(len(array) * train_size):]
    return array_train, array_test


def one_hot(array: np.array) -> np.array:
    # Data types
    unique: np.array
    inverse: np.array
    onehot: np.array

    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot
