'''
My random stuff... sometimes useful, most often not...
'''

from __future__ import division
from numpy import * 

def largestEig(A, its=10):
    "Finds largest eigenpair of matrix using power iterations"

    D = A.size**.5  # get size of input matrix

    v = random.rand(D)  # random starting guess

    for i in range(its):    # power iterations
        v = dot(A,v)
        v = v*linalg.norm(v)**-1
    
    theta = linalg.norm(dot(A,v))
    
    return [theta, v]

D = 100

B = random.rand(D,D)

print largestEig(B)
print linalg.eig(B)
