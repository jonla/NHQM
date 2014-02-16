'''
Created on 15 feb 2014

@author: David Lidberg
'''
import numpy as np


def jacobi_solver(A,b,N):
    '''Solves the system Ax=b with N iterations using the Jacobi
     iteration method'''
    iterations=N
    matrix_size=len(b)
    if (len(b) == len(A) == len(np.transpose(A))):
          
        
        D=np.eye(matrix_size)*np.diag(A)
        R=A-D
        x=np.random.random((matrix_size,1))
        for i in range(iterations):
            x=np.dot(np.linalg.inv(D),(b-np.dot(R,x)))
        return x
    else:
        print "Dimension error"

def offdiag_max(A):
    '''Finds largest off-diagonal element, which determines plane of rotation'''
    A_res=A-np.diag(A)*np.eye(len(A))
    index=np.unravel_index(np.argmax(abs(A_res)),A_res.shape)
    return index

def given_gen(A,index):
    '''Generates a Given rotation matrix in the rotation plane of index'''
    i=np.max(index)
    j=np.min(index)
    if A[i,i]==A[j,j]:
        theta=np.pi/4
    else:
        theta=0.5*np.arctan(2*A[i,j]/(A[j,j]-A[i,i]))
    G=np.eye(len(A))
    c=np.cos(theta)
    s=np.sin(theta)
    G[i,i]=c
    G[j,j]=c
    G[i,j]=s
    G[j,i]=-s
    
    return G

def jacobi_eigensolver(A,N):
    '''Finds eigenvalues of symmetric matrix A with N iterations using Jacobi
    rotation method'''
    tol=0.000001
    for i in range(N):
        ind=offdiag_max(A)
        if abs(A[ind])<tol:
            break
            return A
        else:
        
            given=given_gen(A,ind)
            A=np.transpose(given)*A*given
        
    return A
    
        
    
    





