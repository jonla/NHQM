'''
Created on 27 feb 2014

@author: David Lidberg
'''
import numpy as np
import matplotlib.pyplot as plt


def JD(A,v0,search,tol=0.00001):
    
    '''
    Finds eigenpair for a matrix A with an initial guess v0. 
    The parameter 'search' can either be 'state' or 'energy'.
    In state-mode the algorithm finds the eigenvector of A with
    largest overlap with v0. If search = energy the algorithm identifies
    the eigenvalue of A closest to the value of v0'*A*v0.
    '''
    
     
    
    
    
    
    
    
    
    '''Initialize'''
    theta, V, M, res, iterations = initialize(A,v0)
    theta_init = theta
    theta_hist = np.zeros((iterations,1))
    res_hist = np.zeros((iterations,1))
    I=np.eye(len(v0))    
    count=0
    u=V
    
    
    
    for i in range(iterations):
        theta_hist[0] = theta
        res_hist[0] = np.linalg.norm(res)
        
        
        #print "norm residual", np.linalg.norm(res)
                
        '''Checks for convergence. If not, solves the Jacobi
        orthogonal completion correction for t'''
        if np.linalg.norm(res) < tol:
            
            return theta, u, theta_hist, res_hist, count+1
        
        else:
            t, is_zero = OCC_solve(A,u,theta,res)
            
        
        '''Orthogonalize and normalize t'''  
         
        
        t=modgramschmidt(t,V)
        
        v=t/np.linalg.norm(t)
        
        
        '''Expands search space'''
        V=np.hstack((V,v)) 
        #orthogonality_V = np.dot(np.transpose(V[:,[0]]),V[:,[1]])
        #print "orthogonality of V", orthogonality_V
        
        '''Constructs the M matrix'''     
        M=np.hstack((M,np.zeros((i+1,1))))
        M=np.vstack((M,np.zeros((1,i+2))))
        
        for k in range(i+1):
            
            
            M[i+1,k] = np.dot(np.transpose(V[:,[k]]),np.dot(A,V[:,[i+1]]))
            
            M[k,i+1] = np.dot(np.transpose(V[:,[i+1]]),np.dot(A,V[:,[k]]))
            
          
        M[i+1,i+1] =np.dot(np.transpose(V[:,[i+1]]),np.dot(A,V[:,[i+1]]))
        
        
        
        
        
        '''Finds eigenpair close to guess and builds residue'''
        if search == "state":
            eig_M, vec_M = largest_overlap(M,i)
            
        else:
            if search == "energy":
                eig_M, vec_M = closest_eig(M,theta_init)
            
        
        #print "with np:", eig_M_numpy
        s=vec_M/np.linalg.norm(vec_M)
        
        u=np.dot(V,s)
        
                
        theta=eig_M
        #print "eig_M", eig_M
        
        
        res=np.dot(A,u)-theta*u
        
        '''Saves history for plotting'''
        theta_hist[i+1] = theta
        res_hist[i+1] = np.log10(np.linalg.norm(res))
        count=count+1
        print "Count:", count
        
        
        
        
      
            
        
        
'''SUBROUTINES'''       
        
def initialize(A,v0):
    V=v0/np.linalg.norm(v0)
    theta=np.dot(np.transpose(V),np.dot(A,V))
    
    
    res = np.dot(A,V)-theta*V
    M=theta      
    iterations=len(v0)
    
    return theta, V, M, res, iterations
    
    
    

      
def closest_eig(M,init_theta):
    '''Returns the eigenpair closest to init_theta'''
    
    eig_M, vec_M = np.linalg.eig(M)
    
    difference = eig_M-init_theta
    ind=np.argmin(abs(difference))
    closest_eig = eig_M[[ind]]
    closest_vec = vec_M[:,[ind]]
    
    return closest_eig, closest_vec


def largest_eig(M):
    eig_M, vec_M = np.linalg.eig(M)
    ind = np.argmax(eig_M)
    max_eig = eig_M[ind]
    max_vec = vec_M[:,[ind]]
    
    return max_eig, max_vec

def largest_overlap(M,k):
    eig_M, vec_M = np.linalg.eig(M)
    overlap = np.zeros((k+1,1))
    for j in range(k+1):
        overlap[j] = vec_M[0,j]
        
    ind = np.argmax(abs(overlap))
    overlap_eig = eig_M[ind]
    overlap_vec = vec_M[:,[ind]]
    
    return overlap_eig, overlap_vec
    
    
    
def OCC_solve(A,u,theta,res,weight=100): 
    I=np.eye(len(u))
    
    
    OCC=np.dot((I-np.dot(u,np.transpose(u))),np.dot((A-theta*I),(I-np.dot(u,np.transpose(u)))));
           
    OCC_constrained=np.vstack((OCC, weight*np.transpose(u)));
    res_conc=np.vstack((res,0))
            
    x=np.linalg.lstsq(OCC_constrained,-res_conc);
    t=x[0]
    
    
    orthogonal_check=np.dot(np.transpose(u),t);
    return t, orthogonal_check
    
        
def modgramschmidt(tin, V, kappah=0.25):
    t = tin
    
    if np.shape(V)[1] == 1:
        t = t - np.dot(np.transpose(V),t)*V
    else:
        for j in range(np.shape(V)[1]):
            t = t - np.dot(np.transpose(V[:, [j]]), t)* V[:, [j]]
        if np.linalg.norm(t) / np.linalg.norm(tin) < kappah:
            
            for j in range(np.shape(V)[1]):
                t = t -np.dot(np.transpose(V[:, [j]]), t) * V[:, [j]]
    return t  

def reshist_plot(res, count):
    res_hist = res[1:count,:]
    plt.xlabel("Iterations")
    plt.ylabel("Residual logarithmic norm")
    plt.title("Residual history")
    return plt.plot(res_hist)

def thetahist_plot(res, count):
    res_hist = res[1:count,:]
    plt.xlabel("Iterations")
    plt.ylabel("Theta")
    return plt.plot(res_hist)