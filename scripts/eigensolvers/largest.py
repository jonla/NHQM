from __future__ import division
from numpy import *
from matrix import realsymmetric


def largestEig(A, its = 100, tol = 10**-10):
    "Finds largest eigenpair of matrix using power iterations"

    D = A.size**.5  # get size of input matrix

    v = random.rand(D)  # random starting guess

    for i in range(its):    # power iterations
        v = dot(A,v)
        v = v*linalg.norm(v)**-1

    theta = linalg.norm(dot(A,v))

    return theta, v


def testlargestEig():
    k = 100
    its = 100
    D = 100

    A = realsymmetric(k, D)
    eig, vec = linalg.eig(A)
    Eig = sort(eig)
    theta, v = largestEig(A, its)

    print "largestEig: ", theta
    print "linalg.eig: ", Eig[k-1]
