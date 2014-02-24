import numpy as np
from matrix import matrixgen
from power import powersolve
from jacobi_iteration import jacobi_eigensolver
'''
This script is used to test different eigensolvers.
The resaults can be compared using numpys eig function.

'''

k = 100         # Matrix size
TOL = 1.e-1     # Margin of error
D = 100        # Size of diagonal shape
N = 5000        # Number of itterations for Jacobi

A = matrixgen(k, D)

e, w = powersolve(A, TOL)
Jac = jacobi_eigensolver(np.asmatrix(A), N)
Pow = np.sort(e)
print "Computed eigenvalues using Power"
print Pow[0]



print "Computed eigenvalues using Jacobi"
print Jac[0]

eig= np.linalg.eigvals(A)
Eig = np.sort(eig)
print "Computed eigenvalues using eig"
print Eig[0]

print "Standard error Power:", np.std(Eig - Pow)
print "Standard error Jacobi:", np.std(Eig - Jac)
