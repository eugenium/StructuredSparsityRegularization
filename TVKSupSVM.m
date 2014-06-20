%Copyright 2012 Eugene Belilovsky
%eugene.belilovsky@ecp.fr

function [wbest,primalObj,errors]=TVKSupSVM(X, Y,k,lambda,lambda2,D,OptParam,hu,w0)
    if (size(X, 1) ~= size(Y, 1))
        writelnLog(-1, 'incorrect input size. It should size(X, 1) == size(Y, 1) == nr samples');
        return;
    end

    if(~exist('OptParam.OptType','var'))
        OptParam.OptType=0; %Stochastic Descent
    end
    if(~exist('w0','var'))
        % w0 = 2*X'*Y./length(Y);
        w0 = zeros(size(X,2),1);
    end


    if(OptParam.OptType==1)
        %Here we huber smooth TV/L1 and hinge loss
        if(size(X,1)>size(X,2)) % lipschitz constant for gradient
            L = eigs(X'*X,1)/(2*hu);
        else
            L = eigs(X*X',1)/(2*hu);
        end

        % to get standard Huber smothing of absolute loss, set eps to zero
        eps = 0;

        if(size(D,1)>size(D,2)) % lipschitz constant for gradient of squared loss
            L2 = eigs(D'*D,1)/(2*hu);
        else
            L2 = eigs(D*D',1)/(2*hu);
        end
        L2 = L2+L2;

        [wbest,errors] = overlap_nest(@(w)((1/lambda)*HingeLoss(w,X,Y,hu)/size(X,1)      +(lambda2/2)*epsInsensitiveLoss(w,D,zeros(size(D,1),1),eps,hu)),...
                                      @(w)((1/lambda)*gradHingeLoss(w,X,Y,hu)/size(X,1)  +(lambda2/2)*gradEpsInsensitiveLoss(w,D,zeros(size(D,1),1),eps,hu)), 1-lambda2, ...
                                           (1/lambda)*L/size(X,1)                        +(lambda2/2)*L2, w0, k, OptParam.tol,OptParam.epochs,50);
        primalObj=errors(end);

    else
        %Here we do not need to smooth TV but provide a subgradient
        f=@(x,y,w) (1/lambda)*huberLoss(w,x,y,hu)/size(x,1)+(lambda2)*sum(abs(D*w))/2+((1-lambda2)*norm_overlap(w,k).^2)/2;
        gradf=@(x,y,w) gPrimal(x,y,w,k,lambda,lambda2,D,hu);
        [wbest,primalObj,errors]=MiniBatchStochSubGradientDescent(X,Y,f,gradf,OptParam,w0);
    end
end


function g=gPrimal(x,y,w,k,lambda,lambda2,D,hu)
    [~,gKsup]=norm_overlap(w,k);
    gKsup=2*gKsup;
    g=(1/lambda)*gradHuberLoss(w,x,y,hu)/size(x,1)+(lambda2/2)*computeTV1grad(D,w)+((1-lambda2)/2)*gKsup;
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
    g = zeros(size(w));

    ind1=-D*w<=0;
    g = g + sum(D(ind1,:),1)';

    ind2=-D*w>=0;
    g = g - sum(D(ind2,:),1)';
    s=g;

end

function [ind1,ind2] = huberEpsIndneg(w,X,Y,h,eps)
    ind1 = find(Y-X*w <= -eps-h);
    ind2 = find(abs(Y-X*w + eps) <=h);
end

function [ind1, ind2] = huberEpsIndpos(w,X,Y,h,eps)
    ind1 = find(Y - X*w >= eps+h);
    ind2 = find(abs(Y-X*w - eps)<=h);
end
function l = epsInsensitiveLoss(w,X,Y,eps,h)
    [ind1,ind2] = huberEpsIndneg(w,X,Y,h,eps);
    l = 0;
    if(length(ind1)>0)
        l = l + sum(-eps - Y(ind1) + X(ind1,:)*w);
    end
    if(length(ind2)>0)
        l = l + sum((Y(ind2) - X(ind2,:)*w + eps - h).^2)/(4*h);
    end
    [ind1,ind2] = huberEpsIndpos(w,X,Y,h,eps);
    if(length(ind1)>0)
        l = l + sum(Y(ind1) - X(ind1,:)*w - eps);
    end
    if(length(ind2)>0)
        l = l + sum((Y(ind2) - X(ind2,:)*w - eps + h).^2)/(4*h);
    end
end


function g = gradEpsInsensitiveLoss(w,X,Y,eps,h)
    [ind1,ind2] = huberEpsIndneg(w,X,Y,h,eps);
    g = zeros(size(w));
    if(length(ind1)>0)
        g = g + sum(X(ind1,:),1)'; % careful not to sum up if there is only one training sample
    end
    if(length(ind2)>0)
        g = g + X(ind2,:)'*(X(ind2,:)*w - eps + h - Y(ind2))/(2*h);
    end
    [ind1,ind2] = huberEpsIndpos(w,X,Y,h,eps);
    if(length(ind1)>0)
        g = g - sum(X(ind1,:),1)';
    end
    if(length(ind2)>0)
        g = g + X(ind2,:)'*(X(ind2,:)*w + eps - h - Y(ind2))/(2*h);
    end
end