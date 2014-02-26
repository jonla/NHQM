import numpy as np
from numpy import dot
from largest import largestEig
from numpy.linalg import norm
from matrix import realsymmetric


def davidsolver(A, guess, iterations, eps):
    '''
    This function is meant to solve compute the
    largest eigenvalue to A with corresponding eigenvector.

    TODO:
        * Implement the correction equation with suitable
        solving method and preconditioner.
        * Implement a new eigensolving algorithm for M.
    '''
    V, M, theta, r = davidinit(A, guess)
    n = len(A)
    for m in range(iterations):
        t = solvecorrectioneq(A, theta, r, n)
        t = modgramshmidt(t, V)
        V = np.vstack((V, t / norm(t)))
        # print "V: ", V
        Mp = M
        M = np.zeros((m + 2, m + 2))
        M[0:m + 1, 0:m + 1] = Mp
        for i in range(m + 2):
            M[i, m + 1] = dot(V[i, :], dot(A, V[m + 1, :]))
            M[m + 1, i] = dot(V[m + 1, :], dot(A, V[i, :]))
        '''
        We need to implement a better/faster method than power
        iterations in the long run.
        '''
        theta, s = largestEig(M, 100)
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
    '''
    This part is clearly not optimal.
    Here we are trying to use Davidson's preconditioner
    instad of the jacobi-davidson.
    This is an exact, very costly solution
    and should be replaced when we study larger
    matrices.
    '''
    t = dot(np.linalg.inv(np.diag(A) * np.eye(n)
                          - theta * np.eye(n)), r)
    return t


def modgramshmidt(tin, V, kappah=0.25):
    t = tin
    if len(V.shape) == 1:
        t = t - dot(t, V) * V
    else:
        for j in range(len(V)):
            t = t - dot(t, V[j, :]) * V[j, :]
        if norm(t) / norm(tin) < kappah:
            for j in range(len(V)):
                t = t - dot(t, V[j, :]) * V[j, :]
    return t


def davidsontest():
    k = 100                     # Matrix size
    TOL = 1.e-3                 # Margin of error
    D = 100                     # Diagonal shape
    N = 60                      # Iterations

    A = realsymmetric(k, D)

    eig, vec = np.linalg.eig(A)
    Eig = np.sort(eig)

    guess = np.random.rand(k)
    theta, u = davidsolver(A, guess, N, TOL)

    print "Computed largest eigenvalue davidsolver:"
    print "Eigenvalue = ", theta
    # print "Eigenvector:"
    # print u

    print "Computed smallest and largest using eig"
    print Eig[k - 1], ", ", Eig[0]
