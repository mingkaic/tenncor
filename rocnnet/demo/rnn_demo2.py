import sys
import itertools
import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library
import time

# Set the seed for reproducability
np.random.seed(seed=1)

# Create dataset
nb_train = 2000  # Number of training samples
# Addition of 2 n-bit numbers can result in a n+1 bit number
sequence_len = 7  # Length of the binary sequence

def loss(Y, T):
    """Compute the loss at the output."""
    return -np.mean((T * np.log(Y)) + ((1-T) * np.log(1-Y)))

# Show an example input and target
def printSample(x1, x2, t, y=None):
    """Print a sample in a more visual way."""
    x1 = ''.join([str(int(d)) for d in x1])
    x1_r = int(''.join(reversed(x1)), 2)
    x2 = ''.join([str(int(d)) for d in x2])
    x2_r = int(''.join(reversed(x2)), 2)
    t = ''.join([str(int(d[0])) for d in t])
    t_r = int(''.join(reversed(t)), 2)
    if not y is None:
        y = ''.join([str(int(d[0])) for d in y])
    print(f'x1:   {x1:s}   {x1_r:2d}')
    print(f'x2: + {x2:s}   {x2_r:2d}')
    print(f'      -------   --')
    print(f't:  = {t:s}   {t_r:2d}')
    if not y is None:
        print(f'y:  = {y:s}')

def create_dataset(nb_samples, sequence_len):
    """Create a dataset for binary addition and
    return as input, targets."""
    max_int = 2**(sequence_len-1) # Maximum integer that can be added
     # Transform integer in binary format
    format_str = '{:0' + str(sequence_len) + 'b}'
    nb_inputs = 2  # Add 2 binary numbers
    nb_outputs = 1  # Result is 1 binary number
    # Input samples
    X = np.zeros((nb_samples, sequence_len, nb_inputs))
    # Target samples
    T = np.zeros((nb_samples, sequence_len, nb_outputs))
    # Fill up the input and target matrix
    for i in range(nb_samples):
        # Generate random numbers to add
        nb1 = np.random.randint(0, max_int)
        nb2 = np.random.randint(0, max_int)
        # Fill current input and target row.
        # Note that binary numbers are added from right to left,
        #  but our RNN reads from left to right, so reverse the sequence.
        X[i,:,0] = list(
            reversed([int(b) for b in format_str.format(nb1)]))
        X[i,:,1] = list(
            reversed([int(b) for b in format_str.format(nb2)]))
        T[i,:,0] = list(
            reversed([int(b) for b in format_str.format(nb1+nb2)]))
    return X, T

# Create training samples
X_train, T_train = create_dataset(nb_train, sequence_len)
print(f'X_train tensor shape: {X_train.shape}')
print(f'T_train tensor shape: {T_train.shape}')

# Define the linear tensor transformation layer
class Dense(object): # same thing as dense
    """The linear tensor layer applies a linear tensor dot product
    and a bias to its input."""
    def __init__(self, n_in, n_out, tensor_order):
        """Initialse the weight W and bias b parameters."""
        a = np.sqrt(6.0 / (n_in + n_out))
        self.W = (np.random.uniform(-a, a, (n_in, n_out)))
        self.b = (np.zeros((n_out)))

        # Axes summed over in backprop
        self.bpAxes = tuple(range(tensor_order-1))

    def connect(self, X):
        """Perform connect step transformation with the help
        of a tensor product."""
        # Same as: Y[i,j,:] = np.dot(X[i,j,:], self.W) + self.b
        #          (for i,j in X.shape[0:1])
        # Same as: Y = np.einsum('ijk,kl->ijl', X, self.W) + self.b
        return np.tensordot(X, self.W, axes=((-1),(0))) + self.b

    def backward(self, X, gY):
        """Return the gradient of the parmeters and the inputs of
        this layer."""
        # Same as: gW = np.einsum('ijk,ijl->kl', X, gY)
        # Same as: gW += np.dot(X[:,j,:].T, gY[:,j,:])
        #          (for i,j in X.shape[0:1])
        gW = np.tensordot(X, gY, axes=(self.bpAxes, self.bpAxes))
        gB = np.sum(gY, axis=self.bpAxes)
        # Same as: gX = np.einsum('ijk,kl->ijl', gY, self.W.T)
        # Same as: gX[i,j,:] = np.dot(gY[i,j,:], self.W.T)
        #          (for i,j in gY.shape[0:1])
        gX = np.tensordot(gY, self.W.T, axes=((-1),(0)))
        return gX, gW, gB

# Define layer that unfolds the states over time
class RecurrentStateUnfold(object): # dense -> activation_tanh
    """Unfold the recurrent states."""
    def __init__(self, nbStates):
        """Initialse the shared parameters, the inital state and
        state update function."""
        a = np.sqrt(6. / (nbStates * 2))
        self.S0 = np.zeros(nbStates)  # Initial state
        self.linear = Dense(nbStates, nbStates, 2)

    def connect(self, X):
        """Iteratively apply connect step to all states."""
        # State tensor
        S = np.zeros((X.shape[0], X.shape[1]+1, self.linear.W.shape[0]))
        S[:,0,:] = self.S0  # Set initial state
        nbTimesteps = X.shape[1]
        for k in range(nbTimesteps):
            # Update the states iteratively
            S[:,k+1,:] = np.tanh(X[:,k,:] + self.linear.connect(S[:,k,:]))
        return S

    def backward(self, X, S, gY):
        """Return the gradient of the parmeters and the inputs of
        this layer."""
        # Initialise gradient of state outputs
        nbTimesteps = X.shape[1]
        gSk = np.zeros_like(gY[:,0,:])
        # Initialse gradient tensor for state inputs
        gZ = np.zeros_like(X)
        gWSum = np.zeros_like(self.linear.W)  # Initialise weight gradients
        gBSum = np.zeros_like(self.linear.b)  # Initialse bias gradients
        # Propagate the gradients iteratively
        for k in range(nbTimesteps-1, -1, -1):
            # Gradient at state output is gradient from previous state
            #  plus gradient from output
            gSk += gY[:,k,:]
            # Propgate the gradient back through one state

            gZ_tmp = (1.0 - (S[:,k+1,:]**2)) * gSk
            gSk, gW, gB = self.linear.backward(S[:,k,:], gZ_tmp)
            gZ[:,k,:] = gZ_tmp
            gWSum += gW  # Update total weight gradient
            gBSum += gB  # Update total bias gradient
        # Get gradient of initial state over all samples
        gS0 = np.sum(gSk, axis=0)
        return gZ, gWSum, gBSum, gS0

# Define the full network
class RnnBinaryAdder(object):
    """RNN to perform binary addition of 2 numbers."""
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states):
        """Initialse the network layers."""
        # Input layer
        self.tensorInput = Dense(nb_of_inputs, nb_of_states, 3)
        # Recurrent layer
        self.rnnUnfold = RecurrentStateUnfold(nb_of_states)
        # Linear output transform
        self.tensorOutput = Dense(nb_of_states, nb_of_outputs, 3)

    def connect(self, X):
        """Get the output probabilities of input X."""
        _, _, _, Y = self.full_connect(X)
        return Y

    def full_connect(self, X):
        """Perform the connect propagation of input X through all
        layers."""
        # Linear input transformation
        recIn = self.tensorInput.connect(X)
        # connect propagate through time and return states
        S = self.rnnUnfold.connect(recIn)
        # Linear output transformation
        Z = self.tensorOutput.connect(S[:,1:,:])
        Y = 1. / (1. + np.exp(-Z))
        return recIn, S, Z, Y

    def backward(self, X, Y, recIn, S, T):
        """Perform the backward propagation through all layers.
        Input: input samples, network output, intput to recurrent
        layer, states, targets."""
        sequence_len = X.shape[1]
        gZ = (Y - T) / (Y.shape[0] * Y.shape[1])
        gRecOut, gWout, gBout = self.tensorOutput.backward(
            S[:,1:sequence_len+1,:], gZ)
        # Propagate gradient backwards through time
        gRnnIn, gWrec, gBrec, gS0 = self.rnnUnfold.backward(
            recIn, S, gRecOut)
        _, gWin, gBin = self.tensorInput.backward(X, gRnnIn)
        # Return the parameter gradients of: linear output weights,
        #  linear output bias, recursive weights, recursive bias, #
        #  linear input weights, linear input bias, initial state.
        return gWout, gBout, gWrec, gBrec, gWin, gBin, gS0

class RnnTrainer(object):
    def __init__(self, rnn, learning_rate, momentum_term, lmbd, eps):
        self.rnn = rnn

        self.vars = [
            rnn.tensorOutput.W,
            rnn.tensorOutput.b,
            rnn.rnnUnfold.linear.W,
            rnn.rnnUnfold.linear.b,
            rnn.tensorInput.W,
            rnn.tensorInput.b,
            rnn.rnnUnfold.S0,
        ]
        self.momentums = [np.zeros(v.shape) for v in self.vars]
        self.movingAvgSqr = [np.zeros(v.shape) for v in self.vars]

        self.params = (learning_rate, momentum_term, lmbd, eps)

    def train(self, X, T):
        learning_rate, momentum_term, lmbd, eps = self.params

        momentum_tmp = [v * momentum_term for v in self.momentums]
        self.rnn.tensorOutput.W += momentum_tmp[0]
        self.rnn.tensorOutput.b += momentum_tmp[1]
        self.rnn.rnnUnfold.linear.W += momentum_tmp[2]
        self.rnn.rnnUnfold.linear.b += momentum_tmp[3]
        self.rnn.tensorInput.W += momentum_tmp[4]
        self.rnn.tensorInput.b += momentum_tmp[5]
        self.rnn.rnnUnfold.S0 += momentum_tmp[6]

        recIn, S, _, Y = self.rnn.full_connect(X)
        grads = self.rnn.backward(X, Y, recIn, S, T)

        # update moving average
        for i in range(len(self.vars)):
            self.movingAvgSqr[i] = lmbd * self.movingAvgSqr[i] + (1-lmbd) * grads[i]**2

        pGradNorms = [(learning_rate * grad) / np.sqrt(mvsqr) + eps
            for grad, mvsqr in zip(grads, self.movingAvgSqr)]

        # update momentums
        for i in range(len(self.vars)):
            self.momentums[i] = momentum_tmp[i] - pGradNorms[i]

        self.rnn.tensorOutput.W -= pGradNorms[0]
        self.rnn.tensorOutput.b -= pGradNorms[1]
        self.rnn.rnnUnfold.linear.W -= pGradNorms[2]
        self.rnn.rnnUnfold.linear.b -= pGradNorms[3]
        self.rnn.tensorInput.W -= pGradNorms[4]
        self.rnn.tensorInput.b -= pGradNorms[5]
        self.rnn.rnnUnfold.S0 -= pGradNorms[6]

# keep this here to get nice weights
Dense(2, 3, 3)
RecurrentStateUnfold(3)
Dense(3, 1, 3)

# Set hyper-parameters
lmbd = 0.5  # Rmsprop lambda
learning_rate = 0.05  # Learning rate
momentum_term = 0.80  # Momentum term
eps = 1e-6  # Numerical stability term to prevent division by zero
mb_size = 100  # Size of the minibatches (number of samples)

# Create the network
nb_of_states = 3  # Number of states in the recurrent layer
RNN = RnnBinaryAdder(2, 1, nb_of_states)

trainer = RnnTrainer(RNN, learning_rate, momentum_term, lmbd, eps)

# Create a list of minibatch losses to be plotted
ls_of_loss = [
    loss(RNN.connect(X_train[0:mb_size,:,:]), T_train[0:mb_size,:,:])]
# Iterate over some iterations
start = time.time()
for i in range(5):
    # Iterate over all the minibatches
    for mb in range(nb_train // mb_size):
        X_mb = X_train[mb:mb+mb_size,:,:]  # Input minibatch
        T_mb = T_train[mb:mb+mb_size,:,:]  # Target minibatch

        trainer.train(X_mb, T_mb)

        # Add loss to list to plot
        ls_of_loss.append(loss(RNN.connect(X_mb), T_mb))
print('training time: {} seconds'.format(time.time() - start))

# Plot the loss over the iterations
fig = plt.figure(figsize=(5, 3))
plt.plot(ls_of_loss, 'b-')
plt.xlabel('minibatch iteration')
plt.ylabel('$\\xi$', fontsize=15)
plt.title('Decrease of loss over backprop iteration')
plt.xlim(0, 100)
fig.subplots_adjust(bottom=0.2)
plt.show()

# Create test samples
nb_test = 5
Xtest, Ttest = create_dataset(nb_test, sequence_len)
# Push test data through network
Y = np.around(RNN.connect(Xtest))

# Print out all test examples
for i in range(Xtest.shape[0]):
    printSample(Xtest[i,:,0], Xtest[i,:,1], Ttest[i,:,:], Y[i,:,:])
    print('')
