import numpy as np
from matrix import matrixgen
from power import powersolve
from jacobi_iteration import jacobi_eigensolver
from davidson import davidsolver
'''
This script is used to test different eigensolvers.
The resaults can be compared using numpys eig function.

'''

k = 4         # Matrix size
TOL = 1.e-1     # Margin of error
D = 100        # Size of diagonal shape
N = 5000        # Number of itterations for Jacobi

D = 6           # Size of diagonal shape
N = 4        # Number of iterations


# A = matrixgen(k, D)
A = np.array([[1, 2, 3, 4],
              [0, 4, 2, 1],
              [0, 0, 2, 1],
              [0, 0, 0, 0.5]])

e, w = powersolve(A, TOL)
Jac = jacobi_eigensolver(np.asmatrix(A), N)
Pow = np.sort(e)
print "Computed eigenvalues using Power"
print Pow[0]



print "Computed eigenvalues using Jacobi"
print Jac[0]

# e, w = powersolve(A, TOL)
# Pow = np.sort(e)
# print "Cumputed eigenvalues using Power"
# print Pow

# D = jacobi_eigensolver(np.asmatrix(A), N)
# Jac = np.sort(np.diag(D))
# print "Computed eigenvalues using Jacobi"
# print Jac


eig, vec= np.linalg.eig(A)
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

# print "Standard error Power:", np.std(Eig - Pow)
# print "Standard error Jacobi:", np.std(Eig - Jac)
