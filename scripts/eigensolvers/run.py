import numpy as np
from test import matrixgen
from power import powersolve
from jacobi_iteration import jacobi_eigensolver
'''
This script is used to test different eigensolvers.
The resaults can be compared using numpys eig function.

'''

k = 100         # Matrix size
TOL = 1.e-1     # Margin of error
D = 200           # Size of diagonal shape
N = 5000        # Number of itterations for Jacobi

A = matrixgen(k, D)

e, w = powersolve(A, TOL)
Pow = np.sort(e)
print "Cumputed eigenvalues using Power"
print Pow

D = jacobi_eigensolver(np.asmatrix(A), N)
Jac = np.sort(np.diag(D))
print "Computed eigenvalues using Jacobi"
print Jac

eig, vec = np.linalg.eig(A)
Eig = np.sort(eig)
print "Computed eigenvalues using eig"
print Eig

print "Standard error Power:", np.std(Eig - Pow)
print "Standard error Jacobi:", np.std(Eig - Jac)
