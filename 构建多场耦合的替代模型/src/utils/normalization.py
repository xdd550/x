import numpy as np

def minmax_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def standard_norm(x):
    return (x - np.mean(x)) / np.std(x)