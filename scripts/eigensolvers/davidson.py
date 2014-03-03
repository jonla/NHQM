import numpy as np
from numpy import dot
# from largest import largestEig
from numpy.linalg import norm
from matrix import realsymmetric
import time


def davidsolver(A, guess, iterations, eps):
    '''
    This function is meant to solve compute the
    largest eigenvalue to A with corresponding eigenvector.

    TODO:
        * Implement the correction equation with suitable
        solving method and preconditioner.
        * Implement a new eigensolving algorithm for M.
    '''
    # Timing stuff
    startsolver = time.time()
    timeCEQ = 0

    V, M, theta, r, iterations = davidinit(A, guess)
    
    theta1 = theta
    n = len(A)
    u = V
    for m in range(iterations):
        startCEQ = time.time()
        t = solvecorrectioneq(A, u, theta, r, n)
        timeCEQ = timeCEQ + time.time() - startCEQ
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
        #thetai = abs(evals - theta1).argmin()
        thetai = evals.argmax()
        theta = evals[thetai]
        s = evecs[:, [thetai]]
        u = dot(V, s)
        r = dot(A, u) - theta * u
        f = norm(r)
        print "Iteration:", m, " Theta:", theta, " norm(r):", f
        if f < eps:
            print "Total time:", (time.time() - startsolver)
            print "time on CEQ:", (timeCEQ / (time.time() - startsolver))
            return theta, u
    print "Total time:", (time.time() - startsolver)
    print "time on CEQ:", (timeCEQ / (time.time() - startsolver))
    return theta, u


def davidinit(A, guess):
    iterations = len(guess)
    V = guess / norm(guess)
    theta = dot(V.T, dot(A, V))
    M = theta
    r = dot(A, V) - theta * V
    return V, M, theta, r, iterations


def solvecorrectioneq(A, u, theta, r, n):
    '''
    This is a very costly solution
    and should be replaced when we study larger
    matrices. The method we should use relies on finding a suitable
    preconditioner to (A-theta*I) and then the iterativ method GMRES.
    '''
    # Davidson's suggested correction equation
    # t = np.linalg.solve(np.diag(A) * np.eye(n) - theta * np.eye(n), -r)

    # The JD correction equation
    K = dot(np.eye(n) - np.outer(u, u), dot(A - theta * np.eye(n),
                                            np.eye(n) - np.outer(u, u)))
    K = np.vstack([K, 100 * u.T])   # lstsq weight 100 for u * t = 0
    t = np.linalg.lstsq(K, np.vstack([-r, 0]))[0]
    return t


def modgramshmidt(tin, V, kappah=0.25):
    t = tin

    #if len(V.shape) == 1 or len(V[0,:]) == 1:
     #   t = t - dot(t, V) * V

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
    k = 5                     # Matrix size
    TOL = 1.e-3                 # Margin of error
    D = 5                     # Diagonal shape
    N = 45                      # Iterations

    A = realsymmetric(k, D)
    # A = np.array([[1, 2, -3, 4, 5],
    #               [2, 3, 4, 5, 6],
    #               [-3, 4, 5, -6, 7],
    #               [4, 5, -6, 7, -8],
    #               [5, 6, 7, -8, 9]])
    eig, vec = np.linalg.eig(A)
    Eig = np.sort(eig)
    target = 3 # = eig.argmax()
    eigmax = eig[target]
    vmax = vec[:, [target]]

    # guess = np.random.rand(k, 1)
    # guess = np.ones((k, 1))
    guess = vmax + 0.07 * np.ones((k, 1))
    guess = guess / norm(guess)
    print "Matrix size:", k
    print "Target eigenvalue:", eigmax
    print "Guess'*Vmax:", dot(guess.T, vmax)
    theta1 = dot(guess.T, dot(A, guess))
    print "Theta 1:", theta1
    theta, u = davidsolver(A, guess, N, TOL)

    print "RESULTING EIGENVALUE:", theta
    neari = abs(eig - theta1).argmin()
    neareig = eig[neari]
    nearvec = vec[:,[neari]]
    print "Nearest eigenvalue:", neareig
    print "Guess*nearest:", dot(nearvec.T, guess)
    print "Computed smallest and largest using eig:"
    print Eig[-1], ", ", Eig[0]
