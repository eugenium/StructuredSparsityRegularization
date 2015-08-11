# -*- coding: utf-8 -*-
"""
This module implements key functions related to the k-support norm including computation of the proximal operators of ||x||_ksp^2,||x||_ksp, and Indicator of the ksuport norm ball

"""
#Author: Eugene Belilovsky
import numpy as np
from scipy.linalg import toeplitz
import time

class Ksupport:
    def __init__(self):
        self.init_s_=None
        self.end_s_=None
    def findR(self,w,k):
        d = w.shape[0];
        beta= np.r_[np.Inf,np.sort(np.abs(w))[::-1]];
        
        temp = np.sum(beta[k:d+1]);
        for r in range(0,k):
          if ( (temp >= (r+1.0)*beta[k-r]) and (temp < (r+1.0)*beta[k-r-1])or r==(k-1) ):
            break;
          else:
            temp = temp + beta[k-r-1];

        return r,temp,beta[1:]

    def SolveQuartic(self,z,cumz,cumz_sq,k,r,l,lam,closedForm=True,eps=10**(-15),debug=False,Dual=False):
        #Solves for beta the quartic equation sum(z, 1 to k-r-1) + 1/(r+1) *(sum())**2=lam^2
        n=l-k+r+1
#        D=np.sum(z[1:(k-r)]**2)
#        T=np.sum(z[(k-r):(l+1)])
        D=cumz_sq[k-r-1]
        T=cumz[l]-cumz[k-r-1]
        
#        if(np.abs(D2-D)>10**(-5) or np.abs(T-T2)>10**(-5)):
#            print 'Stop it bro'
        if(Dual):           
            a0=(n)**2*(-lam**2+D)+(r+1)*(T**2)
            a1=-2*lam**2*n*(r+1+n)+2*(T**2*(r+1)+D*(r+1)*n)
            a2=T**2*(r+1)-lam**2*(n**2+4*(r+1)*n+(r+1)**2)+D*(r+1)**2
            a3=-lam**2*(2*(r+1)*n+2*(r+1)**2)
            a4=-lam**2*(r+1)**2
        else:
            a0=-(lam**2)*(n**2)
            a1=-lam**2*(2*n)*(n+r+1)
            a2=-lam**2*(n**2+4*(r+1)*n+(r+1)**2)+D*n**2+(r+1)*T**2
            a3=-2*lam**2*((r+1)*n+(r+1)**2)+2*D*(r+1)*n+2*(r+1)*T**2
            a4=(r+1)**2*(-lam**2+D)+(r+1)*(T**2)
        
        coeff=[a4,a3,a2,a1,a0]
        coeff=np.array(coeff)/np.max(np.abs(np.array(coeff)))
        if(np.isnan(np.sum(coeff))):
            coeff=[a4,a3,a2,a1,a0]
        if(not closedForm):
            beta=np.sort(np.roots(coeff))
            return beta,T
        #print coeff
        #print beta
      #  p=a3/a4;q=a2/a4;r=a1/a4;s=a0/a4
      #  (x1,x2,x3,x4)=Carpenter(p,q,r,s)
        x1=-1;x2=-1;x3=-1;x4=-1
        #a,b,c,d,e=[a4,a3,a2,a1,a0]
        a,b,c,d,e=coeff#[1,a3/a4,a2/a4,a1/a4,a0/a4]
        del_0=c**2-3.0*b*d+12*a*e
        del_1=2*c**3-9*b*c*d+27*b**2*e+27*a*d**2-72*a*c*e
        diff=del_1**2-4.0*del_0**3
        if(np.abs(diff)<eps):
            diff=0.0
        determ=diff/(-27.0)
        P=(8.0*a*c-3.0*b**2)
        D=64*a**3*e-16*a**2*c**2+16*a*b**2*c-16*a**2*b*d-3*b**4
        p=P/(8*a**2)
        q=(b**3-4*a*b*c+8*a**2*d)/(8*a**3)
        
        if(determ>0):
            if(P<0 and D<0):
                phi=np.arccos(del_1/(2*np.sqrt(del_0**3)))
                S=np.sqrt(-2*p/3.0+2*np.sqrt(del_0)*np.cos(phi/3)/(3.0*a))/2
            else:
                return np.array([x1,x2,x3,x4]),T
        else:
            cubearg=(del_1+np.sqrt(diff))/2
            if(cubearg>0):
                Q=(cubearg)**(1.0/3.0)
            elif(cubearg<0):
                Q=-(np.abs(cubearg))**(1.0/3.0)
            else:
                Q=0
            S=np.sqrt(-2*p/3.0+(Q+del_0/Q)/(3.0*a))/2 #TODO fix this when it is zero that is causing most of the NaN cases
            
        sqr1=-4.0*S**2-2*p+(q/S)
        sqr2=-4.0*S**2-2*p-(q/S)
        if(sqr1>=0):
            x1=-b/(4.0*a)-S-np.sqrt(sqr1)/2.0
            x2=-b/(4.0*a)-S+np.sqrt(sqr1)/2.0
        if(sqr2>=0):
            x3=-b/(4.0*a)+S-np.sqrt(sqr2)/2.0
            x4=-b/(4.0*a)+S+np.sqrt(sqr2)/2.0
        
        beta=np.array([x1,x2,x3,x4])
        if(np.isnan(np.sum(beta)) or np.isinf(np.sum(beta)) or np.isnan(sqr1) or np.isnan(sqr2)):
            #TODO this is a bandaid 
           # print 'Warning NaN detected in quartic solver, switching to estimate'
          #  print coeff
            beta=np.sort(np.roots(coeff))
            #print beta
            return beta,T
            
        #b=max(beta)
        if(debug and np.sum(beta>0)>1):
                    print 'WARNING SHOULD only be one positive beta'
                    print determ>0,(P<0 and D<0),sqr1>=0,sqr2>=0,b,T/(l-k+r*(b+1.0)+b+1.0)#,max(np.sort(beta))-max(np.sort(np.roots(coeff)))
#       # time.sleep(0.1)
#        print max(np.sort(beta)),max(np.sort(np.roots(coeff)))
#        time.sleep(0.001)
        return beta,T
    def ProxBallAssist(self,beta,z,k,l,r,w,lam,debug=False,check=False,Dual=False):
        sortOrd=np.argsort(np.abs(w))
        D=np.sum(z[1:(k-r)]**2)
        T=np.sum(z[(k-r):(l+1)])
        compVal=T/(l-k+r*(beta+1.0)+beta+1.0)
        if(z[k-r-1]/(1.0+beta)>compVal and z[l]>compVal and z[l+1]<=compVal):
            if(debug):
                print 'CONDITIONS SATISFIED',l,r,beta
                D=np.sum(z[1:(k-r)]**2)
                n1=l-k+r+1
                Tq=np.sum(z[(k-r):(l+1)]-T/(n1+beta*(r+1)))
                print np.sqrt(D*(beta/(beta+1))+(1.0/(r+1))*Tq**2)
        else:
            if(debug):
                print 'Warning condition not satisfied'
                check=True

        q2=np.zeros(w.shape)
        q=np.zeros(w.shape)
        if(Dual):
            d=len(w)
            q2[0:(k-r-1)]=z[1:k-r]/(beta+1.0)
            if(beta==0):
                q2[(k-r-1):l]=np.sqrt((lam**2-D)/(r+1))            
            else:
                q2[(k-r-1):l]=compVal
            q2[l:]=z[(l+1):(d+1)]
        else:
            q2[0:(k-r-1)]=(beta/(beta+1))*z[1:k-r]
            q2[k-r-1:l]=z[k-r:l+1]-compVal
        q2=q2[::-1]
        q[sortOrd]=q2
        q=np.sign(w)*q
        #q=self.Prox(w,1.0/np.sqrt(beta),k)
        if(debug or check):
            print 'Located at:',self.dualf(q,k)
            print q.dot(w-q),self.dualf(w-q,k)*lam
            print 'beta:', beta
            print r,compVal,l
#                ProjDist=np.linalg.norm(q-w)**2
        return q
    def ProxBall(self,w,lam,k,eps=10**(-25),debug=False,Optimized=False,IgnoreInside=True,Dual=False):
        if(lam<=1.0):
            eps=lam*eps
        if(Dual):
            if(self.dualf(w,k)<=lam+eps and IgnoreInside):
                return w
        else:
            if(self.f(w,k)<=lam+eps and IgnoreInside):
                return w
        q=None
        d=w.shape[0]
        z=np.r_[np.Inf,np.sort(np.abs(w))[::-1],-np.Inf]
        cumz=np.r_[0,np.cumsum(z[1:d+1])]
        cumz_sq=np.r_[0,np.cumsum(z[1:d+1]**2)]
#        sortOrd=np.argsort(np.abs(w))


        if(not Optimized):
            begin_s=0
            end_s=k-1
        else:
            begin_s=self.init_s_
            end_s=self.end_s_

        while(begin_s<=end_s):
       # for s in range(0,k):
            s=(begin_s+end_s)//2
            curr_s=s
            r=k-s-1
            found=False
            
            begin_l=k
            end_l=d

            
            while(begin_l<=end_l):
            #for l in range(k,d+1):
                l=(begin_l+end_l)//2
                curr_l=l               
                betas,T=self.SolveQuartic(z,cumz,cumz_sq,k,r,l,lam,Dual=Dual)
                betas=np.real(betas[~np.iscomplex(betas)])

                if(np.sum(betas>0)>1):
                    beta=max(betas)
                    compVal=T/(l-k+r*(beta+1.0)+beta+1)
                elif(np.sum(betas>0)<1):
                    beta=0.0
                    compVal=0
                else:
                    beta=max(betas)
                    compVal=T/(l-k+r*(beta+1.0)+beta+1)

                if(debug):
                    print 'l:',l,'r:',r,'Val:',compVal,'z[k-r-1]:',z[k-r-1]/(beta+1),'z[k-r]:',z[k-r]/(beta+1),'z[l]:',z[l],'z[l+1]:',z[l+1] 

                
                if(z[l]<compVal):
                    end_l=curr_l-1
                elif(z[l+1]>compVal):
                    begin_l=curr_l+1
                else:
                    #found the correct l
                    if(debug):
                        print  'FOUND S, l:',l,'r:',r,'beta:',beta,'Val:',compVal,'z[k-r-1]:',z[k-r-1]/(beta+1),'z[k-r]:',z[k-r]/(beta+1),'z[l]:',z[l],'z[l+1]:',z[l+1],'u_s:',end_s,'s:',curr_s,'l_s:',begin_s 
                        time.sleep(0.01)
                    found=True
                    break
            if(not found):
                begin_s=curr_s+1
 #               begin_r=curr_r
  #              curr_r=(curr_r+1+end_r)//2    
            elif((z[s]/(1.0+beta))>compVal):

                if(debug):
                    print 'FOUND R, l:',l,'s:',s,'beta:',beta,'Val:',compVal,'z[k-r-1]:',z[k-r-1]/(beta+1),'z[k-r]:',z[k-r]/(beta+1),'z[l]:',z[l],'z[l+1]:',z[l+1],'u_s:',end_s,'s:',curr_s,'l_s:',begin_s 
                    time.sleep(1)                
                q=self.ProxBallAssist(beta,z,k,l,r,w,lam,Dual=Dual)
                
                if(Optimized):
                    self.init_s_=begin_s
                    self.end_s_=end_s
                begin_s=curr_s+1
            else:
                end_s=curr_s-1
                
#            if((end_s-begin_s)<2):
#                    break
        if(q is None):
            if(Optimized):
                q=self.ProxBall(self,w,lam,k,eps=eps,debug=False,Optimized=False,Dual=Dual)
            else:
               # q=self.ProxBallAssist(beta,z,k,l,r,w,lam)
                betas,T=self.SolveQuartic(z,cumz,cumz_sq,k,r,l,lam,Dual=Dual)
                
                if(debug):
                  print 'Not Found using, l:',l,'s:',s,'beta:',beta,'Val:',compVal,'z[k-r-1]:',z[k-r-1]/(beta+1),'z[k-r]:',z[k-r]/(beta+1),'z[l]:',z[l],'z[l+1]:',z[l+1],'u_s:',end_s,'s:',curr_s,'l_s:',begin_s 
                return w #temporary workaround
        if(debug):
            print 'FINISHED'
        if(np.abs(self.f(q,k)-lam)>eps and debug):
            print 'Warning projection did not finish',self.f(q,k)-lam
        return q     

    def Prox(self,w,lam,k,a=0,b=1,eps=10**(-15),debug=False):
        #Algorithm from mcdonald
        #TODO Verify this algorithm again
        w=np.squeeze(w)
        d=w.shape[0]
        abw=np.abs(w).astype(float)
        lowerBnd=(lam)/abw
        upperBnd=(lam+1)/abw
        zerod=abw<eps
        Ordered=np.sort(np.hstack((lowerBnd,upperBnd)))
        
        #binary search
        st=0
        en=2*d-1
        curr=d
        
        found=False
        while st<en:
            
            alpha=Ordered[curr]
            S=np.sum((alpha>upperBnd) & (~zerod) ).astype(float)
            S=S+np.sum(alpha*np.abs(w[(alpha<=upperBnd) & (alpha>=lowerBnd) & (~zerod)])-lam)
              
            if debug:
                print ' S: ',str(S),' k: ',str(k),' st: ',str(st),' curr: ',str(curr),' en: ',str(en)
            if(st+1==en):
                break
            if(S>k):
                #Stopping condition
                en=curr
                curr=np.floor((curr+st)/2).astype(int)    
            elif(S<k):
                st=curr
                curr=np.floor((curr+en)/2).astype(int)
            else:
                found=True
                break
        if st>en:
            print 'Danger will robinson danger'
        
        if not found:
            alpha_low=Ordered[st]
            alpha_hi=Ordered[en]
            S_low=np.sum((alpha_low>upperBnd) & (~zerod) ).astype(float)+np.sum(alpha_low*np.abs(w[(alpha_low<=upperBnd) & (alpha_low>=lowerBnd) & (~zerod)])-lam)
            S_high=np.sum((alpha_hi>upperBnd) & (~zerod) ).astype(float)+np.sum(alpha_hi*np.abs(w[(alpha_hi<=upperBnd) & (alpha_hi>=lowerBnd) & (~zerod)])-lam)  
            alpha=np.interp(k, np.array([S_low,S_high]),np.array([alpha_low,alpha_hi]))
        
        theta=np.zeros(d)
        theta[(alpha<=upperBnd) & (alpha>=lowerBnd) & (~zerod)]=alpha*np.abs(w[(alpha<=upperBnd) & (alpha>=lowerBnd) & (~zerod)])-lam
        theta[(alpha>upperBnd) & (~zerod)]=1
        x=(theta*w)/(theta+lam)
        return x
    def gradf(self,w,k,squared=True):
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
        
        if(not squared):
            #compute the gradient of the non squared norm
            normw = np.sqrt( np.dot(beta[0:k-r-1],beta[0:k-r-1]) + np.power(temp,2)/(r+1) )
            if(normw==0):
                alpha=0
            else:
                alpha=alpha/(2*normw)
        return alpha
    def f(self,w,k):
        #Ksupport
        (r,temp,beta)=self.findR(w,k)
        normw = np.sqrt( np.linalg.norm(beta[0:k-r-1])**2 + np.power(temp,2)/(r+1) );
        return normw
    def dualf(self,a,k):
        #dual norm
        return np.sqrt(np.sum(np.sort(np.abs(a))[::-1][0:k]**2))
    def graddualf(self,w,k):
        #Gradient of Ksupport Dual Norm
        alpha=np.zeros(w.shape)
        d = w.shape[0];
        ind= np.flipud(np.argsort(np.abs(w),0));
        
        alpha[0:k] = np.sort(np.abs(w))[::-1][0:k]/self.dualf(w,k)
        alpha[k:d] = 0
        alpha = np.transpose(alpha)
        rev=np.argsort(ind,0);
        alpha = np.sign(w)* alpha[np.transpose(rev)];
        
        return alpha


def TestKsupCalculation():
    print 'test case for Ksupport class'
    KS=Ksupport();
    t=np.transpose(np.power(10,range(1,6)))
    (r,temp,beta)= KS.findR(t,3)

    print r
    print KS.gradf(t,3)
    print KS.f(t,3)
    print KS.Prox(t,1,3) 
    print KS.Prox(w,1,3)
def TestProx():
    print('Testing Prox Operator')
    w=np.random.randn(10)#np.transpose(np.power(10,range(1,10)))#*np.random.randn(10)
    w=np.sort(np.abs(w))[::-1]
    KS=Ksupport();
    lam=1.1
    k=8
    p= KS.ProxBall(w,lam,k)
    print w
    print p
    print KS.f(p,k)
    #the projection,p, of w should satisfy lam*||w-p||_ksp*=<p,x-p>
    print KS.dualf(w-p,k)*lam
    print p.dot(w-p)
    
    p=KS.ProxBall(w,lam,k,Dual=True)
    print w
    print p
    print KS.dualf(p,k)
    #the projection,p, of w should satisfy lam*||w-p||_ksp*=<p,x-p>
    print KS.f(w-p,k)*lam
    print p.dot(w-p)
    
def main():
    print 'Test gradient of dual'
    KS=Ksupport();
    w=np.random.randn(10)#np.transpose(np.power(10,range(1,10)))#*np.random.randn(10)
    #w=np.sort(np.abs(w))[::-1]
    print w
    print KS.graddualf(w,8)
    print TestProx()

if __name__ == "__main__":
    main()