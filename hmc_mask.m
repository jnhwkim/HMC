function f_mask = hmc_mask( q, size, f_size, sigma )
%HMC_MASK Summary of this function goes here
%   Detailed explanation goes here
%   f_size must be odd if size is even.

f_mask = zeros(size(1)+f_size-1, size(2)+f_size-1);
offset = floor(f_size/2);
f_mask(q(2):q(2)+f_size-1,q(1):q(1)+f_size-1) = ...
    fspecial('gaussian', [f_size f_size], sigma);
f_mask = f_mask(offset+1:end-offset,offset+1:end-offset);

end

