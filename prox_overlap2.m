% copyright 2012 Andreas Argyriou
% GPL License http://www.gnu.org/copyleft/gpl.html

function [ x] = prox_overlap2( v, k, L)

% Compute prox_f(v) for f = 1/(2L) ||.||^2 
% and ||.|| the k overlap norm

d = length(v);
Lambda=sqrt(1/L);
sort('ascend');
end
