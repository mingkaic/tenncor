import numpy as np

def conv_forward(X, kernel):
    '''
    The forward computation for a convolution function

    Arguments:
    X -- output activations of the previous layer, numpy array of shape (n_H_prev, n_W_prev) assuming input channels = 1
    kernel -- Weights, numpy array of size (f, f) assuming number of filters = 1

    Returns:
    H -- conv output, numpy array of size (n_H, n_W)
    cache -- cache of values needed for conv_backward() function
    '''

    # Retrieving dimensions from X's shape
    (n_H_prev, n_W_prev) = X.shape

    # Retrieving dimensions from kernel's shape
    (f, f) = kernel.shape

    # Compute the output dimensions assuming no padding and stride = 1
    n_H = n_H_prev - f + 1
    n_W = n_W_prev - f + 1

    # Initialize the output H with zeros
    H = np.zeros((n_H, n_W))

    # Looping over vertical(h) and horizontal(w) axis of output volume
    for h in range(n_H):
        for w in range(n_W):
            x_slice = X[h:h+f, w:w+f]
            H[h,w] = np.sum(x_slice * kernel)

    # Saving information in 'cache' for backprop
    cache = (X, kernel)

    return H, cache

def conv_backward(doutput, cache):
    '''
    The backward computation for a convolution function

    Arguments:
    doutput -- gradient of the cost with respect to output of the conv layer (H), numpy array of shape (n_H, n_W) assuming channels = 1
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dX -- gradient of the cost with respect to input of the conv layer (X), numpy array of shape (n_H_prev, n_W_prev) assuming channels = 1
    dkernel -- gradient of the cost with respect to the weights of the conv layer (kernel), numpy array of shape (f,f) assuming single filter
    '''

    # Retrieving information from the "cache"
    (X, kernel) = cache

    # Retrieving dimensions from kernel's shape
    (f1, f2) = kernel.shape

    # Retrieving dimensions from dH's shape
    (n_H, n_W) = doutput.shape

    # Initializing dX, dkernel with the correct shapes
    dX = np.zeros(X.shape)
    dkernel = np.zeros(kernel.shape)

    # Looping over vertical(h) and horizontal(w) axis of the output
    for h in range(n_H):
        for w in range(n_W):
            dX[h:h+f1, w:w+f2] += kernel * doutput[h,w]
            dkernel += X[h:h+f1, w:w+f2] * doutput[h,w]

    return dX, dkernel

X = np.array([
    [2, 4, 6],
    [3, 5, 7],
    [8, 9, 11],
])

kernel = np.array([
    [1, 2],
    [3, 4],
])

output, cache = conv_forward(X, kernel)
print(output)

doutput = np.array([
    [7, 11],
    [13, 17],
])

dX, dkernel = conv_backward(doutput, cache)

print(dX)
print(dkernel)
