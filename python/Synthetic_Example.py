# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:49:10 2014

@author: eugene
Copyright 2012 Eugene Belilovsky
eugene.belilovsky@ecp.fr

"""
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import scipy as sp
import itertools
from SolveTVK.ksuptv_solver import tvksp_solver

def Spearman_S(W):
    folds=len(W)
    SpearAll=[]
    for i,j in itertools.combinations(range(folds),2):
        SpearAll.append(sp.stats.spearmanr(W[i],W[j])[0])
    return SpearAll
def SupportRecovery(ground,c,N=50,eps=10**(-12)):
    evalAt=np.linspace(0+eps,np.max(np.abs(c))-eps,N)
    support_true=np.abs(ground)>0
    Precision=np.zeros(N)
    Recall=np.zeros(N)
    for i in range(N):
        val=evalAt[i]
        support=np.abs(c)>val
        TP=float(np.sum( np.logical_and(support_true,support)))
        FP=float(np.sum(np.logical_and(~support_true,support)))
        FN=float(np.sum(np.logical_and(support_true,~support)))
        try:
            Precision[i]=(TP/(TP+FP))
            Recall[i]=(TP/(TP+FN))
        except :
            continue
    RecovPerf=np.trapz(np.flipud(Precision),np.flipud(Recall))
    return Precision,Recall,RecovPerf
    
def genCorr(weights,samples):
    Y=np.random.randint(2,size=samples)
    Y[Y==0]=-1
    X=np.outer(Y,weights);
    X=X+np.random.randn(X.shape[0],X.shape[1])*50;
    return (X,Y)

def ModelSelectAndRetrain(Xtrain,Ytrain,Xval,Yval,alphas,KChoices,ratios,mask,debug=True):
    BestAccuracy=0    
    init=None

    #Search for best validation parameters
    for alpha in alphas:
        for k in KChoices:
            for ratio in ratios:
                max_iter=2000
                prox_max_iter=250
                tol=1e-3
                w,_,init=tvksp_solver(Xtrain,Ytrain,alpha,ratio,k,mask=mask,init=init,loss="logistic",verbose=0,max_iter=max_iter,tol=tol,prox_max_iter=prox_max_iter)
                init=dict()
                init['w']=w
#                KTVS=LF.KtvSVM(D,huber=0.00001,lambda1=1.0/C,lambda2=lam2,k=k)
#                (w,primal)=MiniBatchStochSubGradientDescent(Xtrain,Ytrain,KTVS,w0,alpha0=np.power(10.0,5),N=150,eta=2,tol=np.power(10.0,-6),compareEach=1000,epochs=10000)
                currAccuracy=float(np.sum(np.sign(np.dot(Xval,w[:-1])-w[-1])==Yval))/len(Yval)
                if(currAccuracy>BestAccuracy):
                    best_k=k
                    best_alpha=alpha
                    best_ratio=ratio
                    wbest=w
                    BestAccuracy=currAccuracy
                    if(debug):
                        print 'Found better validation set solution of ',str(BestAccuracy*100), '% at k:',str(k), ' and alpha: ',str(alpha), ' and ratio: ', str(ratio)
                        #time.sleep(5)
    max_iter=2000
    prox_max_iter=400
    tol=1e-5
    w,_,init=tvksp_solver(Xtrain,Ytrain,best_alpha,best_ratio,best_k,mask=mask,init=init,loss="logistic",verbose=0,max_iter=max_iter,tol=tol,prox_max_iter=prox_max_iter)
    return wbest
    
def plotAsst(plt,title,wbest,height=25,width=25):
    vmax = np.abs(wbest).max()
    plt.imshow(wbest.reshape((height,width),order='F'),interpolation="nearest",vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def PerformOneTest(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,weights,mask,ToPlot=False):
    
    alphas=[1000,100,10,1,0.1]
    KChoices=np.array([5,25,45,65,85,200])
    ratios=[0.9,0.8,0.5,0.1]
    result=dict()
    W=dict()
    fig=plt.gcf()
    fig.set_size_inches(11.5,8.5)
    if(ToPlot):
        plt.subplot(3,3,1)
        plotAsst(plt,'GroundTruth',weights)
    

        
    wbest= ModelSelectAndRetrain(Xtrain,Ytrain,Xval,Yval,alphas,[Xtrain.shape[1]],[1],mask)
    inter=wbest[-1]
    wbest=wbest[:-1]
    TestAccuracy=float(np.sum(np.sign(np.dot(Xtest,wbest)+inter)==Ytest))/len(Ytest)
    print 'Best accuracy L2: ',str(100*TestAccuracy), '%'
    result['L2']=TestAccuracy
    W['L2']=wbest    
    if(ToPlot):
        plt.subplot(3,3,2)
        plotAsst(plt,'L2 %.2f%%'%(100*TestAccuracy),wbest)
    
    wbest= ModelSelectAndRetrain(Xtrain,Ytrain,Xval,Yval,alphas,[1],[1],mask)
    inter=wbest[-1]
    wbest=wbest[:-1]
    TestAccuracy=float(np.sum(np.sign(np.dot(Xtest,wbest)+inter)==Ytest))/len(Ytest)
    result['L1']=TestAccuracy
    W['L1']=wbest  
    print 'Best accuracy L1: ',str(100*TestAccuracy), '%'
    if(ToPlot):
        plt.subplot(3,3,3)
        plotAsst(plt,'L1 %.2f%%'%(100*TestAccuracy),wbest)
    
    wbest= ModelSelectAndRetrain(Xtrain,Ytrain,Xval,Yval,alphas,KChoices,[1],mask)
    inter=wbest[-1]
    wbest=wbest[:-1]
    TestAccuracy=float(np.sum(np.sign(np.dot(Xtest,wbest)+inter)==Ytest))/len(Ytest)
    result['KSup']=TestAccuracy
    W['KSup']=wbest  
    print 'Best accuracy KSup: ',str(100*TestAccuracy), '%'
    if(ToPlot):
        plt.subplot(3,3,4)
        plotAsst(plt,'Ksup %.2f%%'%(100*TestAccuracy),wbest)
    
#    wbest= ModelSelectAndRetrain(Xtrain,Ytrain,Xval,Yval,CS,[1],[1],D)
#    TestAccuracy=float(np.sum(np.sign(np.dot(Xtest,wbest))==Ytest))/len(Ytest)
#    print 'Best accuracy TV: ',str(100*TestAccuracy), '%'
#    plt.subplot(2,3,4)
#    plotAsst(plt,'TV',wbest)
#    
    wbest= ModelSelectAndRetrain(Xtrain,Ytrain,Xval,Yval,alphas,[1],ratios,mask)
    inter=wbest[-1]
    wbest=wbest[:-1]
    TestAccuracy=float(np.sum(np.sign(np.dot(Xtest,wbest)+inter)==Ytest))/len(Ytest)
    result['TV_L1']=TestAccuracy
    W['TV_L1']=wbest  
    print 'Best accuracy TV_L1: ',str(100*TestAccuracy), '%'
    if(ToPlot):
        plt.subplot(3,3,5)
        plotAsst(plt,'TV+L1 %.2f%%'%(100*TestAccuracy),wbest)
    
        
    wbest= ModelSelectAndRetrain(Xtrain,Ytrain,Xval,Yval,alphas,KChoices,ratios,mask)
    inter=wbest[-1]
    wbest=wbest[:-1]
    TestAccuracy=float(np.sum(np.sign(np.dot(Xtest,wbest)+inter)==Ytest))/len(Ytest)
    result['KsupTV']=TestAccuracy 
    W['KsupTV']=wbest  
    print 'Best accuracy KsupTV: ',str(100*TestAccuracy), '%'
    if(ToPlot):
        plt.subplot(3,3,7)
        plotAsst(plt,'Ksup/TV %.2f%%'%(100*TestAccuracy),wbest)
    
    return result,W


np.random.seed(0)
height=25
width=25

 
mask=np.ones((height,width),dtype='bool')
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

(Xtrain,Ytrain)=genCorr(weights.ravel(order='F'),150)
(Xval,Yval)=genCorr(weights.ravel(order='F'),100)
(Xtest,Ytest)=genCorr(weights.ravel(order='F'),1000)
Xval=Xval-Xtrain.mean(axis=0)
Xtest=Xtest-Xtrain.mean(axis=0)


alphas=[1000,100,10,1,0.1]
KChoices=np.array([5,25,45,65,85,200])
ratios=[0.9,0.8,0.5,0.1]
wbest= ModelSelectAndRetrain(Xtrain,Ytrain,Xval,Yval,alphas,KChoices,ratios,mask)
inter=wbest[-1]
wbest=wbest[:-1]
TestAccuracy=float(np.sum(np.sign(np.dot(Xtest,wbest)+inter)==Ytest))/len(Ytest)

print 'Best accuracy KsupTV: ',str(100*TestAccuracy), '%'
plt.subplot(1,3,1)
plotAsst(plt,'Original',weights)
plt.subplot(1,3,2)
plotAsst(plt,'Ksup/TV %.2f%%'%(100*TestAccuracy),wbest)
Precision,Recall,RecovPerf=SupportRecovery(weights.ravel(order='F'),wbest,N=50,eps=10**(-12))    
plt.subplot(1,3,3)
plt.plot(Recall,Precision);
plt.xlabel('Recall')
plt.ylabel('Precision')
