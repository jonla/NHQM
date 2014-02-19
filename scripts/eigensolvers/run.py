import numpy as np
from matrix import matrixgen
from power import powersolve
from jacobi_iteration import jacobi_eigensolver
from davidson import davidsolver
'''
This script is used to test different eigensolvers.
The resaults can be compared using numpys eig function.

'''

k = 50         # Matrix size
TOL = 1.e-1     # Margin of error
D = 6           # Size of diagonal shape
N = 500        # Number of iterations

A = matrixgen(k, D)

# e, w = powersolve(A, TOL)
# Pow = np.sort(e)
# print "Cumputed eigenvalues using Power"
# print Pow

# D = jacobi_eigensolver(np.asmatrix(A), N)
# Jac = np.sort(np.diag(D))
# print "Computed eigenvalues using Jacobi"
# print Jac

guess = np.random.rand(len(A))
theta, u = davidsolver(A, guess, N, TOL)
print "Computed largest eigenpair using davidsolver:"
print "Eigenvalue = ", theta
print "Eigenvector:"
print u

eig, vec = np.linalg.eig(A)
Eig = np.sort(eig)
print "Computed eigenvalues using eig"
print Eig

# print "Standard error Power:", np.std(Eig - Pow)
# print "Standard error Jacobi:", np.std(Eig - Jac)
