'''
Created on 15 feb 2014

@author: David Lidberg
'''
import numpy as np
from timing import progress

def jacobi_solver(A,b,N):
    '''Solves the system Ax=b with N iterations using the Jacobi
     iteration method'''
    iterations=N
    matrix_size=len(b)
    
    if (len(b) == len(A) == len(np.transpose(A))):


        D=np.eye(matrix_size)*np.diag(A)
        R=A-D
        x=np.zeros(len(b))
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

def jacobi_eigensolver(Ain,N):
    '''Finds eigenvalues of symmetric matrix A with N iterations using Jacobi
    rotation method'''
    A=np.asarray(Ain)
    tol=0.000001
    for i in progress(range(N)):
        ind=offdiag_max(A)
        if abs(A[ind])<tol:
            break
            return A
        else:

            given=given_gen(A,ind)
            A=np.transpose(given)*A*given
    return np.sort(np.diag(A))

def jacobi_rot(Ain,dom):
    A=np.asmatrix(Ain)
    dominance=0
    ind=offdiag_max(A)
    while (dominance<dom):
        given=given_gen(A,ind)
        A=np.transpose(given)*A*given
        ind=offdiag_max(A)
        dominance=abs(A[0,0]/A[ind])
        
    return A
        




