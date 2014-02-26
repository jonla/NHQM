import numpy as np
from matrix import matrixgen
from power import powersolve
from jacobi_iteration import jacobi_eigensolver
from davidson import davidsolver
from jacobi_iteration import jacobi_rot
from JOCC import JOCC
import timeit
'''
This script is old and should be phased out. Any testing should be done as
a separeate script in each method file.
'''

k = 4         # Matrix size
TOL = 1.e-1     # Margin of error
D = 100        # Size of diagonal shape
N = 5000        # Number of itterations for Jacobi

A = matrixgen(k, D)

# e, w = powersolve(A, TOL)
# Pow = np.sort(e)
# print "Cumputed eigenvalues using Power"
# print Pow


# D = jacobi_eigensolver(np.asmatrix(A), N)
# Jac = np.sort(np.diag(D))
# print "Computed eigenvalues using Jacobi"
# print Jac


eig, vec = np.linalg.eig(A)
Eig = np.sort(eig)



print "Computed eigenvalues using eig"
print Eig[0]

guess = vec[1, :] + 0.1 * vec[2, :] + 0.2 * vec[3, :]
theta, u = davidsolver(A, guess, N, TOL)


print "Computed largest eigenpair using davidsolver:"
print "Eigenvalue = ", theta
print "Eigenvector:"
print u

print "Computed eigenvectors using eig"
print vec



guess = vec[1, :] + 0.1 * vec[2, :] + 0.2 * vec[3, :]
theta, u = davidsolver(A, guess, N, TOL)

print "Computed largest eigenpair using davidsolver:"
print "Eigenvalue = ", theta
# print "Eigenvector:"
# print u

print "Computed eigenvectors using eig"
print Eig[k - 1]


# print "Standard error Power:", np.std(Eig - Pow)
# print "Standard error Jacobi:", np.std(Eig - Jac)
