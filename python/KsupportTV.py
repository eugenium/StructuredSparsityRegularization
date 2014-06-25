# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:05:16 2014

@author: eugene
Copyright 2012 Eugene Belilovsky
eugene.belilovsky@ecp.fr

"""
import numpy as np
from scipy.linalg import toeplitz
from scipy import sparse
      
class TV:
    def __init__(self,D=None):
        self.D=sparse.csr_matrix(D);
        if (D is not None):
            self.p=np.sum(D,0)
        else:
            self.p=None
            
    def gradf(self,w,D=None):
        if(D is None):
            return self.gradfOptimized(w)
            
        g=np.zeros(w.shape)
        D_w=-np.dot(D,w)
        Logical=D_w<=0
        g=g+np.sum(D[Logical,:],0)
        g=g-np.sum(D[~Logical,:],0)
        return g
        
    def gradfOptimized(self,w):
        #requires precomputing
        Log=-self.D.dot(w)<=0
        g=2*self.D[Log,:].sum(0)-self.p
        return np.array(g)
        
    def f(self,w,D):
        return sum(np.abs(np.dot(D,w)))
class Ksupport:
    def findR(self,w,k):
        d = w.shape[0];
        beta= np.flipud(np.sort(np.abs(w),0));
        
        k2=k-1;
        
        temp = np.sum(beta[k2:d]);
        found = False;
        for r in range(0,k-2):
          if ( (temp >= (r+1)*beta[k2-r]) and (temp < (r+1)*beta[k2-r-1]) ):
            found = True;
            break;
          else:
            temp = temp + beta[k2-r-1];
     

        if (not found):
          r=k-1;
        return r,temp,beta
    def gradf(self,w,k):
        #Gradient of Ksupport squared
        alpha=np.zeros(w.shape)
        d = w.shape[0];
        (r,temp,beta)=self.findR(w,k)
        ind= np.flipud(np.argsort(np.abs(w),0));
        
        k2=k-1
        alpha[0:(k-r-1)] = beta[0:(k-r-1)];
        alpha[k2-r:d] = temp / (r+1);
        alpha = np.transpose(alpha)
        rev=np.argsort(ind,0);
        alpha = np.sign(w)* alpha[np.transpose(rev)];
        return alpha
    def f(self,w,k):
        #Ksupport
        (r,temp,beta)=self.findR(w,k)
        normw = np.sqrt( np.dot(beta[0:k-r-1],beta[0:k-r-1]) + np.power(temp,2)/(r+1) );
        return normw

def main():
    # parse command line options
    print 'test case for Ksupport class'
    KS=Ksupport();
    t=np.transpose(np.power(10,range(1,6)))
    (r,temp,beta)= KS.findR(t,3)

    print r
    print KS.gradf(t,3)
    print KS.f(t,3)
    
    print('Testing TV class')
    #Create the D matrix
    D=toeplitz(np.hstack([1,np.zeros(5)]),np.hstack([1,-1, np.zeros(4)]))
    D=np.c_[D,np.zeros(6)]
    D[5,6]=-1
    w=np.random.randn(7)
    TVV=TV()
    TVOpt=TV(D)
    print w
    print TVV.f(w,D)
    print TVV.gradf(w,D)
    print TVOpt.f(w,D)
    print TVOpt.gradf(w)       
if __name__ == "__main__":
    main()