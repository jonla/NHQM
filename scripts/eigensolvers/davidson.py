import numpy as np
from numpy import dot
import matplotlib.pyplot as plt
# from largest import largestEig
from numpy.linalg import norm
from matrix import *
import time


def davidsolver(A, guess, iterations, eps):
    '''
    This function is meant to compute the eigenpair with eigenvector
    close to the initial guess.

    TODO:
        * Modify the correction equation to use a suitable
        solver (GMRES) with a preconditioner (?).
        * Maybe implement a new eigensolving algorithm for M.
    '''
    # Timing stuff
    startsolver = time.time()
    timeCEQ = 0

    V, M, theta, r, iterations = davidinit(A, guess)

    # theta1 = theta
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

        # thetai = abs(evals - theta1).argmin()
        # thetai = abs(evals).argmax()
        thetai = abs(evecs[0, :]).argmax()

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

    # K = A - theta * np.eye(n)
    # Kinv = np.linalg.inv(K)
    # Ktilde = dot(np.eye(n) - np.outer(u, u), dot(K,
    #                                          np.eye(n) - np.outer(u, u)))
    # uhat = dot(Kinv, u)
    # mu = dot(u.T, uhat)
    # rhat = dot(Kinv, r)
    # rtilde = rhat - dot(u.T,rhat) / mu * uhat

    # # Here we should apply the Krylov sover GMRES:
    # v = np.linalg.solve(dot(np.linalg.inv(Ktilde), A), -r)

    # y = dot(A - theta * np.eye(n), v)
    # yhat = dot(Kinv, y)
    # z = yhat - dot(u, yhat) / mu * uhat
    # Normalized z -> t ?
    return t


def modgramshmidt(tin, V, kappah=0.25):
    t = tin

    #if len(V.shape) == 1 or len(V[0,:]) == 1:
     #   t = t - dot(t, V) * V

    if len(V[1]) == 1:
        t = t - dot(t.T, V) * V

    else:
        for j in range(len(V.T)):
            t = t - dot(t.T, V[:, [j]]) * V[:, [j]]
        if norm(t) / norm(tin) < kappah:
            for j in range(len(V.T)):
                t = t - dot(t.T, V[:, [j]]) * V[:, [j]]
    return t / norm(t)


def davidsontest(k=100, TOL=1.e-3, N=45, hermitian=True, real=True,
                 target=18, guessoffset=0.15):

    # k = 100                     # Matrix size
    # TOL = 1.e-3                 # Margin of error
    D = 1000                     # Diagonal shape
    # N = 45                      # Iterations

    if hermitian:
        A = realsymmetric(k, D) if real else complexhermitian(k, d)
    else:
        A = complexsymmetric(k, D)

    eig, vec = np.linalg.eig(A)
    Eig = np.sort(eig)
    # target = 18  # = eig.argmax()
    eigtarget = eig[target]
    vtarget = vec[:, [target]]

    # guess = np.random.rand(k, 1)
    # guess = np.ones((k, 1))

    guess = vtarget + guessoffset * np.ones((k, 1))
    guess = guess / norm(guess)
    theta1 = dot(guess.T, dot(A, guess))
    print "Matrix size:", k
    print "Target eigenvalue:", eigtarget
    print "Guess'*Vtarget:", dot(guess.T, vtarget)
    print "Theta 1:", theta1

    theta, u = davidsolver(A, guess, N, TOL)
    neari = abs(eig - theta1).argmin()
    neareig = eig[neari]
    nearvec = vec[:, [neari]]
    print "RESULTING EIGENVALUE:", theta
    print "Nearest eigenvalue to Theta 1:", neareig
    print "Guess*nearest:", dot(nearvec.T, guess)
    print "Computed smallest and largest using eig:"
    print Eig[-1], ", ", Eig[0]
