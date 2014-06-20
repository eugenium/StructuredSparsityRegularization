%Copyright 2012 Eugene Belilovsky
%eugene.belilovsky@ecp.fr

function [PrimalObj,ksupErr,tvErr] = ComputePrimalNoDelay(X,Y,lambda,k,lambdaTV,D,w)
% Computes the primalObjective for the k-support tv norm regularized hinge
% loss

NumSamples=size(X,1);


hingeErr=(1/lambda)*hingeLoss(w,X,Y)/NumSamples;
tvErr=(lambdaTV/2)*sum(abs(D*w));
ksupErr=((1-lambdaTV)/2)*(norm_overlap(w,k).^2);
regErr=tvErr+ksupErr;


PrimalObj=hingeErr+regErr;
        
end

function l = hingeLoss(w,X,Y)
    l = sum(max(0,1-Y.*(X*w)));
end

% end of file
