'''
Created on 19 feb 2014

@author: David Lidberg
'''
import numpy as np
from jacobi_iteration import jacobi_solver
'''JOCC method for finding eigenvalue. I think it works now
but as far as I can tell there's no way to predict which eigenvalue
it finds - it alternates between the largest and second largest. I
dont know why.'''

def JOCC(Ain,N):
    A=np.asarray(Ain)
    alpha=A[0,0]
    b=A[1:,0]
    c=A[0,1:]
    F=A[1:,1:]
    z=np.zeros(len(A)-1)
    I=np.eye(len(A)-1)

    for i in range(N):
        theta=alpha+np.dot(np.transpose(c),z)
        z=jacobi_solver((F-theta*I),(-b),z,N)
        
    return theta
'''
It could probaby be modified to return the eigenvector approximation as well,
I think it's z'''
    
            