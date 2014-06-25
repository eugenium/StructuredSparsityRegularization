%Copyright 2012 Eugene Belilovsky
%eugene.belilovsky@ecp.fr
function [wbest,primalObj,errors]=MiniBatchStochSubGradientDescent(X,Y,f,gradf,OptParam,w0)

NumSamples=size(X,1);
features=size(X,2);

if(~exist('w0','var') || isempty(w0))
    w0 = zeros(features,1); % initial point
end

%Initialize the parameters
alpha0=OptParam.alpha0;
N=OptParam.N;
epochs=OptParam.epochs;
eta=OptParam.eta;
alpha=alpha0;
tol=OptParam.tol;
compareEach=OptParam.compareEach;

n_fl=floor(NumSamples/N)*N;
buf=buffer(1:n_fl,N);

if(exist('OptParam.type','var') && OptParam.type==0)
    buf=(1:N)';
end


Le=Inf;
fbest=Inf;
errors=zeros(1,epochs);
w=w0;
for epoch=1:epochs
    %compute objective value at the start of each epoch
    
    e=f(X,Y,w);
    MSE=sum((X*w-Y).^2)/NumSamples;
    Class=sum(sign(X*w)==Y)/NumSamples;

    errors(epoch) = e;
    
    %if we have improved save the best result otherwise go back to the best
    %weight vector and reduce the step size
    if(e>fbest)
        w=wbest;
        alpha=alpha/eta;
        errors(epoch)=fbest;
        e=fbest;
    else
        wbest=w;
        fbest=e;
        if((Le-e)/e<tol)
            break;
        end
        Le=e;
    end

    if(mod(epoch,compareEach)==0)
        display(['The current error is: ' num2str(e) ' MSE and Classification Rate: ' num2str(MSE) ',' num2str(Class)  ' and the current stepsize is ' num2str(alpha)])
    end
    
    %Run through the dataset
    perm=randperm(NumSamples);
    for i=1:size(buf,2)
        samples=perm(buf(:,i));
        % get a sample

        y_cur= Y(samples,:);        
        x_cur=X(samples,:);

        % noisy subgradient calculation

        g=gradf(x_cur,y_cur,w);
        
        if(norm(g)==0)
            break;
        end
        % step size selection
        alpha2 = alpha/norm(g);
        
        % subgradient update
        w = w - alpha2*g;
    end

end
primalObj=errors(end);
end
