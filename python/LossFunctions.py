# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 20:53:08 2014

@author: eugene
Copyright 2012 Eugene Belilovsky
eugene.belilovsky@ecp.fr

"""
import numpy as np
import KsupportTV as KTV
from scipy.linalg import toeplitz
class KtvSVM:
    """Define the objective Function"""

    def __init__(self,D,loss='HuberSVM', penalty='KsupTV',lambda1=1,lambda2=0.1,k=1,huber=0.01):
        self.k=k;
        self.lambda2=lambda2;
        self.lambda1=lambda1;
        self.D=D;
        self.SVMLoss=SVMHuber(huber)
        self.TVReg=KTV.TV(D)
        self.KSup=KTV.Ksupport()
        
    def gradf(self,X,Y,w):
        gLoss=(1/self.lambda1)*self.SVMLoss.gradf(X,Y,w)/X.shape[0]
        gTV=(self.lambda2/2)*self.TVReg.gradf(w)
        gKsup=(1-self.lambda2)*self.KSup.gradf(w,self.k)
        return gLoss+gTV+gKsup
    def f(self,X,Y,w):
        pLoss=(1/self.lambda1)*self.SVMLoss.f(X,Y,w)/X.shape[0]
        pTV=(self.lambda2/2)*self.TVReg.f(w,self.D)
        pKsup=((1-self.lambda2)/2)*np.power(self.KSup.f(w,self.k),2)
        return pLoss+pTV+pKsup



#  
class SVMHuber:
    def __init__(self,huber=0.01):
        self.hu=huber;
        
    def gradf(self,X,Y,w):
        (ind1,ind2,_)=self.huberInd(X,Y,w)
        g=np.zeros(w.shape)
        g=g-np.dot(X[ind1,:].T,Y[ind1]).T
        X_2=X[ind2,:]

        g=g+(np.dot(X_2.T,np.dot(X_2,w)) -(1+self.hu)*np.dot(X_2.T,Y[ind2])).T/(2*self.hu)
       
        return g
        
    def f(self,X,Y,w):
        (ind1,ind2,margin)=self.huberInd(X,Y,w)
        l1= np.sum(1-margin[ind1])
        l=np.sum(np.power((1+self.hu)-margin[ind2],2))/(4*self.hu)
        return l+l1
        
    def huberInd(self,X,Y,w):
        margin=Y*np.dot(X,w)
        ind1=np.where(margin<1-self.hu)[0]
        ind2=np.where(np.abs(1-margin)<=self.hu)[0]
        return (ind1,ind2,margin)
        
def main():

    
    print('Testing SVM class')
    height=3
    width=5
    samples=10
    #Create the D matrix
    D_w=toeplitz(np.hstack([1,np.zeros(width-2)]),np.hstack([1,-1, np.zeros(width-2)]))
    D_h=toeplitz(np.hstack([1,np.zeros(height-2)]),np.hstack([1,-1, np.zeros(height-2)]))
#    D=np.c_[D,np.zeros(width-1)]
#    D[width-2,width-1]=-1
    D2=np.kron(D_w,np.eye(height))
    D3=np.kron(np.eye(width),D_h)
    D=np.r_[D2,D3]
    
    
    w=np.random.randn(height*width)
    X=np.random.randn(samples,height*width)
    Y=np.random.randint(2,size=samples)
    Y[Y==0]=-1
    
    SHuber=SVMHuber(huber=0.5)
    print w
    print X
    print SHuber.gradf(X,Y,w)
    print SHuber.f(X,Y,w)
    
    print('Testing P class')
    KTVS=KtvSVM(D,huber=0.01,lambda1=1,lambda2=0.1,k=1)
    print KTVS.f(X,Y,w)
    print KTVS.gradf(X,Y,w)
      
if __name__ == "__main__":
    main()