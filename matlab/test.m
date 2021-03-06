%Copyright 2012 Eugene Belilovsky
%eugene.belilovsky@ecp.fr

%% Generate synthetic data
rng(0);
height=25;
width=25;


filt=fspecial('gaussian', [5 5], 3);
weights=zeros(25,25);
weights(5,5)=1;
weights(end-5,end-5)=1;
weights=filter2(filt,weights);
%smooth it out
weights=filter2(filt,weights);
weights(weights~=0)=weights(weights~=0)/mean(mean(weights))-1;

[Xtrain,Ytrain] = gendataCorr(250,weights(:));
[Xval,Yval] = gendataCorr(100,weights(:));
[Xtest,Ytest] = gendataCorr(250,weights(:));


%% Setup Model Selection
C=10.^(0:3); %SVM C
%model parameters
KChoices=[5:20:100];
Lambda1=1./C;
Lambda2=[0 0.1 0.5 0.8 1];
D=sparse(GenerateIncidence4Neighbor(height,width));  % Matrix for Total Variation Operator


Nk=length(KChoices);
NL1=length(Lambda1);
NL2=length(Lambda2);


%% Initialize the Optimization Parameters
OptParam.OptType=2; %Use Stochastic Descent
OptParam.N=5; %Number of elements used to get stochastic gradient
OptParam.epochs=100000; % Maximum number of epochs
OptParam.compareEach=1; %Show SGD progress every few iterations
OptParam.tol=1e-5;
OptParam.alpha0=10^3; %initial step size (make this high)
OptParam.eta=2; %decrease the step size by this factor when we do not descent
hu=0.00001; %smoothing parameter used for SVM


%% Perform model selection
BestAccuracy=0;
wbest=[];
for a=1:Nk
    for b=1:NL1
        for c=1:NL2
            k=KChoices(a);
            lambda=Lambda1(b);
            lambda2=Lambda2(c);
            [w,primalObj,errors]=TVKSupSVM(Xtrain, Ytrain,k,lambda,lambda2,D,OptParam,hu);
            currAccuracy=sum(sign(Xval*w)==Yval)/length(Yval);
            if(currAccuracy>BestAccuracy)
                wbest=w;
                BestAccuracy=currAccuracy;
                display(['Found better validation set solution of ' num2str(BestAccuracy*100) '% at k:' num2str(k) ' and lambda: ' num2str(lambda) ' and lambda2: ' num2str(lambda2)]);
            end
        end
    end
end

TestAccuracy=sum(sign(Xtrain*wbest)==Ytrain)/length(Ytrain);
figure; subplot(122); imagesc(reshape(wbest,height,width));
title(['Best accuracy: ' num2str(100*TestAccuracy) '%']);
subplot(121); imagesc(reshape(weights,height,width));