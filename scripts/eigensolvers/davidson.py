import numpy as np
from numpy import dot
from largest import largestEig
from numpy.linalg import norm


def davidsolver(A, guess, iterations, eps):
    '''
    This function is meant to solve compute the
    largest eigenvalue to A with corresponding eigenvector.

    ERRORS:
        The norm of the residual is diverging.
    JL 19/2
    '''
    V, M, theta, r = davidinit(A, guess)
    n = len(A)
    for m in range(iterations):
        t = solvecorrectioneq(A, theta, r, n)
        for j in range(m):
            t = t - dot(dot(t, V[m - 1, :]), V[j, :])
        V = np.vstack((V, t / norm(t)))
        print "V: ", V
        Mp = M
        M = np.zeros((m + 2, m + 2))
        M[0:m + 1, 0:m + 1] = Mp
        for i in range(m + 2):
            M[i, m + 1] = dot(V[i, :], dot(A, V[m + 1, :]))
        print "M: ", M
        [theta, s] = largestEig(M, 10)
        u = dot(np.transpose(V), s)
        r = dot(A, u) - theta * u
        f = norm(r)
        if f < eps:
            return theta, u
    return theta, u


def davidinit(A, guess):
    V = guess / norm(guess)
    theta = dot(np.transpose(V), dot(A, V))
    M = theta
    r = dot(A, V) - theta * V
    return V, M, theta, r


def solvecorrectioneq(A, theta, r, n):
    # This part is clearly not optimal.
    # Here we are trying to use Davidson's preconditioner
    # instad of the jacobi-davidson.
    t = dot(np.linalg.inv(np.diag(A) * np.eye(n)
                          - theta * np.eye(n)), r)
    return t
