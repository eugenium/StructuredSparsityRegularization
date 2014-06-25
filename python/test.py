# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:49:10 2014

@author: eugene
Copyright 2012 Eugene Belilovsky
eugene.belilovsky@ecp.fr

"""
import numpy as np
import LossFunctions as LF
from scipy.signal import convolve2d
from MiniBatchStochSubGradientDescent import MiniBatchStochSubGradientDescent
from scipy.linalg import toeplitz

def genCorr(weights,samples):
    Y=np.random.randint(2,size=samples)
    Y[Y==0]=-1
    X=np.outer(Y,weights);
    X=X+np.random.randn(X.shape[0],X.shape[1])*50;
    return (X,Y)

def main():

    np.random.seed(0)
    height=25
    width=25
    
    #Create the D matrix
    D_w=toeplitz(np.hstack([1,np.zeros(width-2)]),np.hstack([1,-1, np.zeros(width-2)]))
    D_h=toeplitz(np.hstack([1,np.zeros(height-2)]),np.hstack([1,-1, np.zeros(height-2)]))
    D2=np.kron(D_w,np.eye(height))
    D3=np.kron(np.eye(width),D_h)
    D=np.r_[D2,D3]
    
    #copy kernel from matlab to compare directly
    kernel=np.array([[0.0318,0.0375,0.0397,0.0375,0.0318],
     [0.0375,0.0443,0.0469,0.0443,0.0375],
     [0.0397,0.0469,0.0495,0.0469,0.0397],
     [0.0375,0.0443,0.0469,0.0443,0.0375],
     [0.0318,0.0375,0.0397,0.0375,0.0318] ]) 
    
    weights=np.zeros([height,width])
    weights[4,4]=1
    weights[height-5,width-5]=1    
    weights=convolve2d(weights,kernel,mode='same')
    weights=convolve2d(weights,kernel,mode='same')
    weights[weights!=0]=weights[weights!=0]/np.mean(np.mean(weights))-1;
    
    (Xtrain,Ytrain)=genCorr(np.reshape(weights,height*width),250)
    (Xval,Yval)=genCorr(np.reshape(weights,height*width),100)
    (Xtest,Ytest)=genCorr(np.reshape(weights,height*width),250)
    CS=[1,10,100,1000]
    KChoices=[5,25,45,65,85]
    Lamb2=[0,0.1,0.5,0.8,1]
    

    BestAccuracy=0    
    w0=np.zeros(Xtrain.shape[1])    
    #Search for best validation parameters
    for C in CS:
        for k in KChoices:
            for lam2 in Lamb2:
                KTVS=LF.KtvSVM(D,huber=0.00001,lambda1=1.0/C,lambda2=lam2,k=k)
                (w,primal)=MiniBatchStochSubGradientDescent(Xtrain,Ytrain,KTVS,w0,alpha0=np.power(10.0,3),N=25,eta=2,tol=np.power(10.0,-6),compareEach=100,epochs=10000)
                currAccuracy=float(np.sum(np.sign(np.dot(Xval,w))==Yval))/len(Yval)
                if(currAccuracy>BestAccuracy):
                    wbest=w
                    BestAccuracy=currAccuracy
                    print 'Found better validation set solution of ',str(BestAccuracy*100), '% at k:',str(k), ' and lambda: ',str(1.0/C), ' and lambda2: ', str(lam2)
    
    TestAccuracy=float(np.sum(np.sign(np.dot(Xtest,wbest))==Ytest))/len(Ytest)
    print 'Best accuracy: ',str(100*TestAccuracy), '%'
if __name__ == "__main__":
    main()