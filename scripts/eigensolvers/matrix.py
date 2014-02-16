import numpy as np


def matrixgen(k, D):
    ''' Creates a k by k sized symmetric matrix
    with D non-zero elements on each row'''
    A = np.zeros((k, k))
    for i in xrange(k):
        for j in xrange(i + 1):
            if np.abs(i - j) < D:
                A[i, j] = np.random.rand(1)
                A[j, i] = A[i, j]
    return A
