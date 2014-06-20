function [w,pH,primalObj,XH]=kNormSSGDSVMLFullBetter(X, Y, firsts, len, windowSize, OptParam,union,w,type)

if (size(X, 1) ~= size(Y, 1))
    writelnLog(-1, 'incorrect input size. It should size(X, 1) == size(Y, 1) == nr samples');
    return;
end

k=OptParam.k;
lambda=OptParam.lambda;
lambda2=OptParam.lambda2;
alpha0=OptParam.alpha0;
N=OptParam.N;
epochs=OptParam.epochs;
eta=OptParam.eta;
hu=OptParam.huber;
D=OptParam.D;

nrChannels=size(X,2);
NumSamples=size(X,1);
n=len*nrChannels;
if(~exist('w','var') || isempty(w))
    w = 0*randn(n,1); % initial point
end

wbest=w;
fbest=Inf;
errors=zeros(1,1);

%alpha=0.000001;
alpha=alpha0;
tol=OptParam.tol;
compareEach=OptParam.compareEach;

n_fl=floor(NumSamples/N)*N;
buf=buffer(1:n_fl,N);

if(type==0)
    buf=(1:N)';
end

lasts = firsts + len - 1;
Xcache_ = zeros(2*windowSize + 1, NumSamples, nrChannels, len);
for i = 1:size(X, 2)
    for h = 1:(2*windowSize + 1)
        h_ = h - windowSize - 1;
        Xcache_(h, :, i, :) = reshape(X(:, i, (firsts(i) + h_):(lasts(i)  + h_)), NumSamples, len);
    end
end
%Xcache2 = reshape(Xcache_, (2*windowSize + 1) * NumSamples, nrChannels * len);
Yrep = repmat(Y', 2 * windowSize + 1, 1); 
Xcache = reshape(Xcache_, [(2*windowSize + 1), NumSamples, nrChannels * len]);

Le=Inf;
fbest=Inf;
x_cur=zeros(N,nrChannels*len);
XH=zeros(NumSamples,nrChannels*len);
for epoch=1:epochs
    %compute objective value at end of epoch
    samples=1:NumSamples;
    %Compute h and h_hat
    PhiHvalues = reshape(reshape(Xcache(:,samples,:),[(2*windowSize + 1)* NumSamples, nrChannels * len]) * w, 2*windowSize+1, NumSamples).* Yrep(:,samples);
    if (size(PhiHvalues, 1) > 1)
        [posPhi, pH] = max(PhiHvalues);
        [negPhi, nH] = max(-PhiHvalues);
    else
        posPhi = squeeze(PhiHvalues);
        pH = ones(1, size(X, 1));
        negPhi = squeeze(PhiHvalues);
        nH = ones(1, size(X, 1));
    end
    
    
    for s = 1:NumSamples
        XH(s,:)=Xcache(pH(s),samples(s),:)+Xcache(nH(s),samples(s),:);
    end
    
    if(union)
        %         lamb3=union;
        %         e=huberLoss(w,Xtrain,Ytrain,h)+lamb3*max((lambda/2)*nval.^2,(lambda2)*TVval);
    else
        
        %e=(1/lambda)*huberLoss(w,XH,Y,h)/NumSamples+((1-lambda2)*norm_overlap(w,k).^2+(lambda2)*sum(abs(D*w)))/2;
        huberErr=(1/lambda)*huberLoss(w,XH,Y,hu)/NumSamples;
        TVerr=(lambda2)*sum(abs(D*w))/2;
        ksupErr=((1-lambda2)*norm_overlap(w,k).^2)/2;
        regErr=ksupErr+TVerr;
        e=huberErr+regErr;
        MSE=sum((XH*w-Y).^2)/NumSamples;
        Class=sum(sign(XH*w)==Y)/NumSamples;
    end
    %e=e/length(Ytrain);
    errors(epoch) = e;
    if(e>fbest)
        w=wbest;
        alpha=alpha/eta;
        errors(epoch)=fbest;
        e=fbest;
    else
        wbest=w;
        fbest=e;
        if((Le-e)<tol)
            break;
        end
        Le=e;
    end

    if(mod(epoch,compareEach)==0)
        display(['The current error is: ' num2str(e) ', Hinge: ' num2str(huberErr) ', Ksup: ' num2str(ksupErr) ', Tv: ' num2str(TVerr) ' MSE and Classification Rate: ' num2str(MSE) ',' num2str(Class)  ' and the current stepsize is ' num2str(alpha)])
    end
    
    %Run through the dataset
    perm=randperm(NumSamples);
    for i=1:size(buf,2)
        samples=perm(buf(:,i));
        % get a sample
        %x_cur = 2*X(samples,:); %for H=0 the kernel is 2*x
        y_cur= Y(samples,:);
        
        
        %Compute h and h_hat
        PhiHvalues = reshape(reshape(Xcache(:,samples,:),[(2*windowSize + 1)* N, nrChannels * len]) * w, 2*windowSize+1, N).* Yrep(:,samples);
        if (size(PhiHvalues, 1) > 1)
            [posPhi, posH] = max(PhiHvalues);
            [negPhi, negH] = max(-PhiHvalues);
        else
            posPhi = squeeze(PhiHvalues);
            posH = ones(1, size(X, 1));
            negPhi = squeeze(PhiHvalues);
            negH = ones(1, size(X, 1));
        end
        
        
        for s = 1:N
            x_cur(s,:)=Xcache(posH(s),samples(s),:)+Xcache(negH(s),samples(s),:);
        end
        %gLoss=posPhi-negPhi
        %x_cur= 
        % noisy subgradient calculation
        
        gLoss = gradHuberLoss(w,x_cur,y_cur,hu);%2*x_cur'*(x_cur*w) - 2*x_cur'*y_cur; %squared loss gradient
        %gL2=2*w;
        [nval,gKsup]=norm_overlap(w,k);
        %TVval=sum(abs(D*w));
        gTV=computeTV1grad(D,w);
        gKsupSquared=2*gKsup;
        if(union)
            %         lamb3=union;
            %         if(lambda*nval>lambda2*TVval)
            %             g=gLoss+lamb3*lambda*gKsupSquared;
            %         else
            %             g=gLoss+lamb3*lambda2*gTV;
            %         end
        else
            g=(1/lambda)*gLoss/N+((1-lambda2)*gKsupSquared+lambda2*gTV)/2;
        end
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


function [ind1,ind2] = huberInd(w,X,Y,hu)
  margin = Y.*(X*w);
  ind1 = find(margin<1-hu);
  ind2 = find(abs(1-margin)<=hu);
end

function l = huberLoss(w,X,Y,hu)
    [ind1,ind2] = huberInd(w,X,Y,hu);
    l = 0;
    if(~isempty(ind1))
        l = sum(1-Y(ind1).*(X(ind1,:)*w));
    end
    l2 = 0;
    if(~isempty(ind2))
        l2 = sum((1+hu-Y(ind2).*(X(ind2,:)*w)).^2)./(4*hu);
    end
    l = l+l2;
end

function g = gradHuberLoss(w,X,Y,hu)
    [ind1,ind2] = huberInd(w,X,Y,hu);
    g = zeros(size(w));
    if(~isempty(ind1))
        g = g - X(ind1,:)'*Y(ind1);
    end
    if(~isempty(ind2))
        g = g + (X(ind2,:)'*(X(ind2,:)*w) - (1+hu)*X(ind2,:)'*Y(ind2))./(2*hu);
    end
end

function s=computeTV1grad(D,w)
% temp=+(D*w>0);
% temp(temp==0)=-1;
% s=sum(bsxfun(@times,D,temp))';
g = zeros(size(w));

ind1=-D*w<=0;
g = g + sum(D(ind1,:),1)';

ind2=-D*w>=0;

g = g - sum(D(ind2,:),1)';
s=g;

end
