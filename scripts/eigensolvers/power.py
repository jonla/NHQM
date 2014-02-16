import numpy as np
from numpy import dot, abs, transpose
from numpy.linalg import norm
from timing import progress


def powersolve(A, TOL):
    '''
    e, w = powersolve(A, TOL) generates eigenvalues e and
    corresponding eigenvectors w of the matrix A.
    TOL is the margin of error.
    '''
    k = A.shape[0]
    v = np.random.rand(k)  # Starting vector
    eig = np.zeros(k)
    vectors = np.empty((k, k))
    u = dot(A, v) / norm(dot(A, v))
    B = A
    for n in progress(xrange(k)):
        while norm(abs(u) - abs(v)) > TOL:
            u = v
            v = dot(B, u)
            v = v / norm(v)
        eig[n] = dot(transpose(v), dot(B, v))
        vectors[n] = v
        B = dot(dot((np.identity(k) - np.outer(v, v)), B),
                (np.identity(k) - np.outer(v, v)))
        v = np.random.rand(k)
    return eig, vectors
