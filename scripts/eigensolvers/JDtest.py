'''
Created on 27 feb 2014

@author: David Lidberg
'''
import numpy as np


from Jacobi_Davidson import *
from matrix import realsymmetric
N=5
B = realsymmetric(N,N)
A = np.asmatrix([ [0.2202,    0.1117,    0.1519,    0.0432,    0.0712],
    [0.1117,    0.5514,   -0.2560,    0.1716,   -0.0654],
    [0.1519,   -0.2560,   -0.6452,    0.7342,   -0.3321],
    [0.0432,    0.1716,    0.7342,   -0.7026,   -0.1678],
    [0.0712,   -0.0654,   -0.3321,   -0.1678,   -0.0926] ])
print type(A)
print type(B)

eig,vec = np.linalg.eig(A)

guess = np.transpose(np.asmatrix(vec[:,0]))+0.15*np.random.random((N,1))
error= np.linalg.norm(guess-np.transpose(np.asmatrix(vec[:,0])))
guess2 = np.asmatrix(np.random.random((N,1)))
alpha=np.dot(np.transpose(guess),np.dot(A,guess))
print alpha
print "error guess:", error
theta, e=JD(A,guess)


print "JD:", theta
print "Numpy:", eig
