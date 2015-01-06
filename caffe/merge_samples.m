function S = merge_samples( S1, S2, n1, n2 )
%MERGE_SAMPLES Summary of this function goes here
%   Detailed explanation goes here

%% Precondition
assert(size(S1,2) == size(S2,2));
assert(size(S1,1)/n1 == size(S2,1)/n2);

%% Preallocation for a merged matrix.
S = zeros(size(S1,1)+size(S2,2),size(S1,2));

for i = 1 : size(S1,1)/n1
   S((i-1)*(n1+n2)+1:(i-1)*(n1+n2)+n1,:) = S1((i-1)*n1+1:i*n1,:);
   S((i-1)*(n1+n2)+n1+1:i*(n1+n2),:) = S2((i-1)*n2+1:i*n2,:);
end

end

