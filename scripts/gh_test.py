'''
My random stuff... sometimes useful, most often not...
'''

from __future__ import division
from numpy import * 
from eigensolvers.largest import largestEig


D = 100

B = random.rand(D,D)

print largestEig(B)
print linalg.eig(B)
