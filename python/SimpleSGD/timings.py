# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 18:36:45 2014

@author: eugene
"""
import time
import numpy as np
from scipy import sparse
import pandas as pd
from scipy.linalg import toeplitz

height=25
width=25

#Create the D matrix
D_w=toeplitz(np.hstack([1,np.zeros(width-2)]),np.hstack([1,-1, np.zeros(width-2)]))
D_h=toeplitz(np.hstack([1,np.zeros(height-2)]),np.hstack([1,-1, np.zeros(height-2)]))
#    D=np.c_[D,np.zeros(width-1)]
#    D[width-2,width-1]=-1
D2=np.kron(D_w,np.eye(height))
D3=np.kron(np.eye(width),D_h)
D=np.r_[D2,D3]



w=np.random.randn(height*width)
#Ds=sparse.csr_matrix(D)
p=np.sum(D,0)

t = time.time()
for i in range(0,100):
    g=np.zeros(w.shape)
    D_w=-np.dot(D,w)
    Log=D_w<=0
    g=g+np.sum(D[Log,:],0)
    g=g-np.sum(D[~Log,:],0)

elapsed=time.time()-t
print elapsed
g2=g

D=sparse.csr_matrix(D)
t = time.time()
for i in range(0,100):
    #g=
    #g=g+np.sum(sub,0)
    Log=-D.dot(w)<=0
    g=2*D[Log,:].sum(0)-p
    
    #g=g+np.sum(D[Log,:],0).T
    #g=g-np.sum(D[Log,:],0).T

elapsed=time.time()-t
print elapsed
print g==g2