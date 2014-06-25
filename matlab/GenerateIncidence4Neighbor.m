function [incid adj]=GenerateIncidence4Neighbor(h,w)

diagVec1 = repmat([ones(w-1,1); 0],h,1);  %# Make the first diagonal vector
                                          %#   (for horizontal connections)
diagVec1 = diagVec1(1:end-1);             %# Remove the last value
diagVec2 = ones(w*(h-1),1);               %# Make the second diagonal vector
                                          %#   (for vertical connections)
adj = diag(diagVec1,1)+...                %# Add the diagonals to a zero matrix
      diag(diagVec2,w);
adj = adj+adj.';                         %'# Add the matrix to a transposed
incid=+full(adj2inc(adj));
isOne = incid == 1 ;
B = isOne & cumsum(isOne,2) == 1;
incid(incid==1)=-1;
incid(B)=1;
%[~,idx] = max(incid_t(:,any(incid_t)));
%incid(incid==1)=-1;
%incid(idx)=1;