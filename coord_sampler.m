function [ S ] = coord_sampler( X, height, width, samples )
%COORD_SAMPLER Summary of this function goes here
%   Detailed explanation goes here

coord = 3;
S = zeros(size(X,1), samples*coord);

%% find indexes
tic;
for i = 1 : size(X, 1)
    x = zeros(samples, 1);
    y = zeros(samples, 1);
    z = zeros(samples, 1);
    count = 1;
    I = reshape(X(i,:), [height width]);
    sorted = unique(sort(X(i,:)));
    for j = size(sorted, 2) : -1 : 1
        [u,v] = find(I==sorted(j));
        append_size = min(samples-count+1, size(u,1));
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
    a = [x';y';z'];
    S(i,:) = a(:);
    if mod(i, 1000) == 0
        fprintf('*');
    end
    if mod(i, 10000) == 0
        fprintf(' ');toc;
    end
end

