function [PrimalObj,ksupErr,tvErr] = ComputePrimal(X,Y,lambda,k,lambdaTV,D,w,firsts,len,windowSize)
% Computes the primalObjective

NumSamples=size(X,1);
nrChannels=size(X,2);
lasts = firsts + len - 1;
Xcache_ = zeros(2*windowSize + 1, NumSamples, nrChannels, len);
for i = 1:size(X, 2)
    for h = 1:(2*windowSize + 1)
        h_ = h - windowSize - 1;
        Xcache_(h, :, i, :) = reshape(X(:, i, (firsts(i) + h_):(lasts(i)  + h_)), NumSamples, len);
    end
end
Yrep = repmat(Y', 2 * windowSize + 1, 1); 
Xcache = reshape(Xcache_, [(2*windowSize + 1), NumSamples, nrChannels * len]);


XH=computeBestAlign(Xcache,Yrep,w,nrChannels,len);

hingeErr=(1/lambda)*hingeLoss(w,XH,Y)/NumSamples;
tvErr=(lambdaTV/2)*sum(abs(D*w));
ksupErr=((1-lambdaTV)/2)*(norm_overlap(w,k).^2);
regErr=tvErr+ksupErr;


PrimalObj=hingeErr+regErr;
        
end

function l = hingeLoss(w,X,Y)
    l = sum(max(0,1-Y.*(X*w)));
end


function XH=computeBestAlign(Xcache,Yrep,w,nrChannels,len)
windowSize=(size(Xcache,1)-1)/2;
NumSamples=size(Yrep,2);
PhiHvalues = reshape(reshape(Xcache,[(2*windowSize + 1)* NumSamples, nrChannels * len]) * w, 2*windowSize+1, NumSamples).* Yrep;
if (size(PhiHvalues, 1) > 1)
    [posPhi, pH] = max(PhiHvalues);
    [negPhi, nH] = max(-PhiHvalues);
else
    posPhi = squeeze(PhiHvalues);
    pH = ones(1, NumSamples);
    negPhi = squeeze(PhiHvalues);
    nH = ones(1, NumSamples);
end

XH=zeros(size(Xcache,2),size(Xcache,3));
for s=1:NumSamples
    XH(s,:)=Xcache(pH(s),s,:)+Xcache(nH(s),s,:);
end

end

% end of file
