import numpy as np


def zeroimpute_recovery(matrix):
    mask = np.isnan(matrix);
    matrix[mask] = 0.0;
    return matrix;