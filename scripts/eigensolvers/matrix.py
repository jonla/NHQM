import numpy as np


def realsymmetric(k, D):
    ''' Creates a k by k sized real symmetric matrix
    with D non-zero elements on each row'''
    A = np.zeros((k, k))
    
    for i in xrange(k):
        for j in xrange(i + 1):
            if np.abs(i - j) < D:
                A[i, j] = np.random.rand(1)
                A[j, i] = A[i, j]
    return A

def complexsymmetric(k, D):
    ''' Creates a k by k sized symmetric complex matrix
    with D non-zero elements on each row'''
    A = np.zeros((k, k),dtype=complex)
    
    for i in xrange(k):
        for j in xrange(i + 1):
            if np.abs(i - j) < D:
                A[i, j:j+1] = np.random.rand(1) + np.random.rand(1)*1j 
                A[j, i] = A[i, j]
    return A

def matrixgen(k, D):
    ''' Creates a k by k sized symmetric matrix
    with D non-zero elements on each row. This
    old method is left here just to not break stuff
    '''
    A = np.zeros((k, k))
    
    for i in xrange(k):
        for j in xrange(i + 1):
            if np.abs(i - j) < D:
                A[i, j] = np.random.rand(1)
                A[j, i] = A[i, j]
    return A

