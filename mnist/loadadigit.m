function [ X, y ] = loadadigit( type, index )
    if strcmp('train', type)
        X = loadMNISTImages('train-images-idx3-ubyte');
        y = loadMNISTLabels('train-labels-idx1-ubyte');
    else
        X = loadMNISTImages('t10k-images-idx3-ubyte');
        y = loadMNISTLabels('t10k-labels-idx1-ubyte');
    end
    X = X(:,index);
    y = y(index);
end

