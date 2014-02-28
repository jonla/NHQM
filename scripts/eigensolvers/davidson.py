import numpy as np
from numpy import dot
# from largest import largestEig
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
    u = V
    for m in range(iterations):
        t = solvecorrectioneq(A, u, theta, r, n)
        vplus = modgramshmidt(t, V)
        M = np.vstack([np.hstack([M, dot(V.T, dot(A, vplus))]),
                       np.hstack([dot(vplus.T, dot(A, V)),
                                  dot(vplus.T, dot(A, vplus))])])
        V = np.hstack((V, vplus))
        '''
        We need to implement a better/faster method than power
        iterations in the long run.
        '''
        # theta, s = largestEig(M, 100)
        evals, evecs = np.linalg.eig(M)
        theta = evals[0]
        s = evecs[:, [0]]
        u = dot(V, s)
        r = dot(A, u) - theta * u
        f = norm(r)
        print "Iteration:", m, " Theta:", theta, " f:", f
        if f < eps:
            return theta, u
    return theta, u


def davidinit(A, guess):
    V = guess / norm(guess)
    theta = dot(V.T, dot(A, V))
    M = theta
    r = dot(A, V) - theta * V
    return V, M, theta, r


def solvecorrectioneq(A, u, theta, r, n):
    '''
    This part is clearly not optimal.
    Here we are trying to use Davidson's preconditioner
    instad of the jacobi-davidson.
    This is an exact, very costly solution
    and should be replaced when we study larger
    matrices.
    '''
    # t = np.linalg.solve(np.diag(A) * np.eye(n) - theta * np.eye(n), -r)
    K = dot(np.eye(n) - np.outer(u, u), dot(A - theta * np.eye(n),
                                            np.eye(n) - np.outer(u, u)))
    K = np.vstack([K, 100 * u.T])   # lstsq weight 100 for u * t = 0
    t = np.linalg.lstsq(K, np.vstack([-r, 0]))[0]
    return t


def modgramshmidt(tin, V, kappah=0.25):
    t = tin
    if len(V[1]) == 1:
        t = t - dot(t.T, V) * V
    else:
        for j in range(len(V.T)):
            # print "dot(t.T, V[:, [j]]): ", dot(t.T, V[:, [j]])
            t = t - dot(t.T, V[:, [j]]) * V[:, [j]]
            # print "t.shape", t.shape
        if norm(t) / norm(tin) < kappah:
            for j in range(len(V.T)):
                t = t - dot(t.T, V[:, [j]]) * V[:, [j]]
    return t / norm(t)


def davidsontest():
    k = 1000                     # Matrix size
    TOL = 1.e-3                 # Margin of error
    D = 100                     # Diagonal shape
    N = 45                      # Iterations

    A = realsymmetric(k, D)
    # A = np.array([[1, 2, -3, 4, 5],
    #               [2, 3, 4, 5, 6],
    #               [-3, 4, 5, -6, 7],
    #               [4, 5, -6, 7, -8],
    #               [5, 6, 7, -8, 9]])
    eig, vec = np.linalg.eig(A)
    Eig = np.sort(eig)

    # guess = np.random.rand(k, 1)
    # guess = np.ones((k, 1))
    guess = vec[:, [0]] + 0.1 * np.ones((k, 1))
    theta, u = davidsolver(A, guess, N, TOL)

    print "Computed largest eigenvalue davidsolver:"
    print "Eigenvalue = ", theta

    print "Computed smallest and largest using eig"
    print Eig[-1], ", ", Eig[0]

    print "Target eigenvalue:", eig[0]
    print "Guess'*Vmax:", dot((guess / norm(guess)).T, vec[:, [0]])
