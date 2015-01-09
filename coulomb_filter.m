function filter = coulomb_filter( height, width )
%COULOMB_FILTER Summary of this function goes here
%   Detailed explanation goes here

filter = zeros(height*2+1, width*2+1);
for i = 1 : size(filter,1)
    for j = 1 : size(filter,2)
        r = sqrt((height+1-i)^2+(width+1-j)^2);
        if r ~= 0
            filter(i,j) = 1/r;
        end
    end
end
    
end

