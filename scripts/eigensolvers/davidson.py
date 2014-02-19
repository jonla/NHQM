import numpy as np
from numpy import dot
from largest import largestEig


def davidsolver(A, guess, iterations, eps):
    '''
    This function is meant to solve compute the
    largest eigenvalue to A with corresponding eigenvector.

    ERRORS:
        The norm of the residual is diverging.
    JL 19/2
    '''
    t = guess
    v = np.zeros((0, len(t)))
    Mp = np.zeros((0, 0))
    for m in range(iterations):
        M = np.zeros((m + 1, m + 1))
        M[0:m, 0:m] = Mp
        for j in range(m - 1):
            t = t + dot(dot(t, v[m - 1, :]), v[j, :])
        v = np.vstack((v, t / np.linalg.norm(t)))
        for i in range(m + 1):
            M[i, m] = dot(v[i, :], dot(A, v[m, :]))
        Mp = M
        [theta, s] = largestEig(M, 10)
        u = dot(np.transpose(v), s)
        r = dot(A, u) - theta * u
        f = np.linalg.norm(r)
        print "Fel: ", f
        if f < eps:
            return theta, u
        # This part is clearly not optimal.
        # Here we are trying to use Davidson's preconditioner
        # instad of the jacobi-davidson.
        t = dot(np.linalg.inv(np.diag(A) * np.eye(len(A))
                              - theta * np.eye(len(A))), r)
    return theta, u
