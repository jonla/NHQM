'''
Created on 25 feb 2014

@author: David Lidberg
'''
'''
Created on 19 feb 2014

@author: David Lidberg
'''
import numpy as np
from jacobi_iteration import jacobi_solver
from matrix import matrixgen
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
    guess=np.random.random(len(A)-1)
    I=np.eye(len(A)-1)
    z=guess

    for i in range(N):
        theta=alpha+np.dot(np.transpose(c),z)
        z=jacobi_solver((F-theta*I),(-b),N)
    
    return theta, z, guess
'''
It could probaby be modified to return the eigenvector approximation as well,
I think it's z'''

def nearest_diag(A,guess):
    D=np.diag(A)-guess
    for m in range(len(D)):
    
        for n in range(len(D)):
            if abs(D[m]) < abs(D[n]):
                return m
                
        
            
                
    return m
        
        
    
    
A=matrixgen(3,3)


ind=nearest_diag(A,0.5)
print A
print ind
print np.diag(A)-0.5

'''
dim=50
diag_dom=50
iterations=100
Q=matrixgen(dim,diag_dom)
Q[0,0]=Q[0,0]+25
eigp=np.linalg.eigvals(Q)
eig, vec, guess = JOCC(Q,iterations)

print "Eig:", eig
print "python",eigp
'''
            