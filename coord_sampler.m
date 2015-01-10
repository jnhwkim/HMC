function [ S ] = coord_sampler( X, height, width, samples )
%COORD_SAMPLER Summary of this function goes here
%   Detailed explanation goes here

coord = 5;
S = zeros(size(X,1), samples*coord);

%% find indexes
tic;
for i = 1 : size(X, 1)
    x = zeros(samples+1, 1);
    y = zeros(samples+1, 1);
    z = zeros(samples+1, 1);
    count = 1;
    I = reshape(X(i,:), [height width]);
    sorted = unique(sort(X(i,:)));
    for j = size(sorted, 2) : -1 : 1
        [u,v] = find(I==sorted(j));
        append_size = min(samples-count+2, size(u,1));
        x(count:count+append_size-1) = u(1:append_size) / width;
        y(count:count+append_size-1) = v(1:append_size) / height;
        z(count:count+append_size-1) = ...
            I(sub2ind(size(I),u(1:append_size), ...
            v(1:append_size)));
        count = count + append_size;
        if count > samples
           break;
       end
    end
    a = [x(1:samples,1)';y(1:samples,1)';...
         (x(2:samples+1,1)'-x(1:samples)'+1)/2;...
         (y(2:samples+1,1)'-y(1:samples)'+1)/2;...
         z(1:samples,1)'];
    S(i,:) = a(:);
    if mod(i, 1000) == 0
        fprintf('*');
    end
    if mod(i, 10000) == 0
        fprintf(' ');toc;
    end
end

