# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:51:21 2014

@author: eugene
Copyright 2012 Eugene Belilovsky
eugene.belilovsky@ecp.fr

"""


import numpy as np
from numpy.linalg import norm

def MiniBatchStochSubGradientDescent(X,Y,ObjectiveFunc,w0,alpha0=np.power(10.0,3),N=1,eta=2,tol=np.power(10.0,-6),compareEach=50,epochs=10000): 
    NumSamples=X.shape[0]
    
    #TODO figure out how to do this in python if(~exist('w0','var') || isempty(w0))
    #w0 = zeros(features,1); % initial point
    #end
    
    #n_fl=np.floor(NumSamples/N)*N
    
    #TODO maybe add this optioon
    #if(exist('OptParam.type','var') && OptParam.type==0)
    #    buf=(1:N)';
    
    
    
    Le=np.Inf
    fbest=np.Inf
    errors=np.zeros(epochs)
    w=w0
    alpha=alpha0
    wbest=w0
    for epoch in range(1,epochs):
        #compute objective value at the start of each epoch
        
        e=ObjectiveFunc.f(X,Y,w);
        MSE=np.sum(np.power(np.dot(X,w)-Y,2))/NumSamples;
        Class=np.sum(np.sign(np.dot(X,w))==Y)/float(NumSamples);
    
        errors = e;
        
        #if we have improved save the best result otherwise go back to the best
        #weight vector and reduce the step size
        if(e>fbest):
            w=wbest;
            alpha=alpha/eta;
            errors=fbest;
            e=fbest;
        else:
            wbest=w;
            fbest=e;
            if( e==0 or (Le-e)/e<tol):
                break
            
            Le=e;
    
    
        if(np.mod(epoch,compareEach)==0):
            print 'The current error is: '+str(e) +' MSE and Classification Rate: ' +str(MSE) +',' +str(Class) + ' and the current stepsize is ' +str(alpha)
    
        
        #Run through the dataset
        perm=np.random.permutation(NumSamples);
        for i in range(int(np.floor(NumSamples/N))):
            samples=perm[i*N:(i+1)*(N)]
            
    
            y_cur= Y[samples];        
            x_cur=X[samples,:];
    
            # noisy subgradient calculation
    
            g=np.squeeze(ObjectiveFunc.gradf(x_cur,y_cur,w));
            
            if(norm(g)==0):
                break;

            # step size selection
            alpha2 = alpha/norm(g);
            
            # subgradient update
            w = w - alpha2*g;
        
    
    
    primalObj=errors;
    
    return(wbest,primalObj)