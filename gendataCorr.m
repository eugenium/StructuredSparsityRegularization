function [X, Y] = gendataCorr(n,weights)

Y = sign(rand(n,1)-0.5);

X = Y*weights';
X = X+randn(size(X))*50;

end