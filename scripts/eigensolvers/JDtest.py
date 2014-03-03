'''
Created on 27 feb 2014

@author: David Lidberg
'''


import numpy as np
from Jacobi_Davidson import *
from matrix import *



'''
Tests for Jacobi_Davidson.py
ISSUES: Less accurate than the matlab version, can't tell why.
        Sometimes the found state is somehow reversed, as if a
        factor of -1 has appeared.

'''



matrix_size = 100 
which_state = 80
error_size = 0.08
print "Target eigenpair:", which_state

A = realsymmetric(matrix_size,matrix_size)
eig,vec = np.linalg.eig(A)
print "Desired eigenvalue:", eig[which_state]


'''Creates initial vector'''
sought_state = vec[:,[which_state]]
sought_value = eig[which_state]
error = error_size*np.ones((matrix_size,1))
guess = sought_state+error
guess_acc = abs(np.dot(np.transpose(sought_state),guess))
print "Guess accuracy:", guess_acc

'''Runs JD'''
theta, e, theta_hist, res_hist, count = JD(A,guess,"state")
print "Completed iterations:", count
print "Found eigenvalue:", theta

'''Checks overlap'''
overlap = np.zeros((matrix_size,1))
for m in range(matrix_size):
    overlap[m,0] = abs(np.dot(np.transpose(vec[:,[m]]),e))
max_overlap_ind = np.argmax(overlap)
found_overlap = vec[:,[max_overlap_ind]]
print "Found state has largest overlap with eigenvector", max_overlap_ind
print "Overlap:", overlap[max_overlap_ind,0]
    


plt.figure(1)
reshist_plot(res_hist,count)


plt.figure(2)
thetahist_plot(theta_hist,count)

plt.figure(3)
plt.plot(e, label="Found state")
plt.plot(sought_state, label="Desired state", color="red")
plt.legend()
plt.show()
