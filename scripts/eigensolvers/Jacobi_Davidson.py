'''
Created on 27 feb 2014

@author: David Lidberg
'''
import numpy as np



def JD(A,v0,tol=0.001):
    '''This part could be placed in an initializer'''
    A=np.asmatrix(A)
    init_theta=np.dot(np.transpose(v0),np.dot(A,v0))
    V=v0/np.linalg.norm(v0)
     
       
    I=np.eye(len(v0))
    M=init_theta
    count=0
    iterations=len(A)
    for i in range(iterations):
               
        '''Finds eigenpair close to guess and builds residue'''
        
        eig_M, vec_M = extract_eig(M,init_theta)
        
        s=np.asmatrix(vec_M)
        
        u=np.dot(V,s)
        #u=np.asmatrix(u)
        
        
        theta=eig_M
        
        
        res=np.dot(A,u)-theta*u
        
        print "norm residual", np.linalg.norm(res)
                
        '''Checks for convergence. If not, solves the Jacobi
        orthogonal completion correction for t'''
        if np.linalg.norm(res) < tol:
            
            return theta, u
        else:
            t, is_zero = OCC_solve(A,u,theta,res)
            
        
        '''Orthogonalize and normalize t'''  
          
        t=modgramschmidt(t,V)
        v=t/np.linalg.norm(t)
        
        
        '''Expands search space'''
        V=np.hstack((V,v)) 
        
        
        
        '''Constructs the M matrix'''     
        M=np.hstack((M,np.zeros((i+1,1))))
        M=np.vstack((M,np.zeros((1,i+2))))
        
        for k in range(i+1):
            
            
            M[i+1,k] = np.dot(np.transpose(V[:,k]),np.dot(A,V[:,i+1]))
            
            M[k,i+1] = np.dot(np.transpose(V[:,i+1]),np.dot(A,V[:,k]))
            
          
        M[i+1,i+1] =np.dot(np.transpose(V[:,i+1]),np.dot(A,V[:,i+1]))
        
        count=count+1
        
        print "Completed iterations:", count
        
        
        
      
            
        
        
'''SUBROUTINES'''       

      
def extract_eig(M,init_theta):
    '''Returns the eigenpair closest to init_theta'''
    
    eig_M, vec_M = np.linalg.eig(M)
    
    difference = eig_M-init_theta
    ind=np.argmin(abs(difference))
    closest_eig = eig_M[ind]
    closest_vec = np.asmatrix(vec_M[:,ind]/np.linalg.norm(vec_M[:,ind]))
    
    return closest_eig, closest_vec
    
    
def OCC_solve(A,u,theta,res,weight=1000): 
    I=np.eye(len(u))
    
    
    OCC=np.asmatrix(np.dot((I-np.dot(u,np.transpose(u))),np.dot((A-theta*I),(I-np.dot(u,np.transpose(u))))));
     
       
    OCC_constrained=np.vstack((OCC, weight*np.transpose(u)));
    
    res_conc=np.vstack((res,0))
    
        
    x=np.linalg.lstsq(OCC_constrained,res_conc);
    t=x[0]
    
    
    orthogonal_check=np.dot(np.transpose(u),t);
    return t, orthogonal_check
    
        
def modgramschmidt(tin, V, kappah=0.30):
    t = tin
    
    if np.shape(V)[1] == 1:
        t = t - np.asscalar(np.dot(np.transpose(V),t))*V
    else:
        for j in range(np.shape(V)[1]):
            t = t - np.asscalar(np.dot(np.transpose(V[:, j]), t))* V[:, j]
        if np.linalg.norm(t) / np.linalg.norm(tin) < kappah:
            
            for j in range(np.shape(V)[1]):
                t = t -np.asscalar((np.dot(np.transpose(V[:, j]), t))) * V[:, j]
    return t  