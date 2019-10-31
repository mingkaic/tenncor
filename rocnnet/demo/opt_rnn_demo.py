# source: https://peterroelants.github.io/posts/rnn-implementation-part02/
import sys
import functools
import time

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library

# Set the seed for reproducability
np.random.seed(seed=1)

def cross_entropy_loss(Y, T):
    epsilon = 1e-5 # todo: make epsilon padding configurable for certain operators in eteq
    leftY = Y + epsilon
    rightT = 1 - Y + epsilon
    return -(T * tc.log(leftY) + (1-T) * tc.log(rightT))

def loss(Y, T):
    return tc.reduce_mean(cross_entropy_loss(Y, T))

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

class Dense(object):
    def __init__(self, n_in, n_out, tensor_order):
        a = np.sqrt(6.0 / (n_in + n_out))
        W = (np.random.uniform(-a, a, (n_in, n_out)))
        self.weight = eteq.variable(W)
        self.bias = eteq.Variable([n_out])

        self.bpAxes = tuple(range(tensor_order-1))

    def connect(self, X):
        out = tc.contract(X, self.weight)
        out += tc.best_extend(self.bias, out.shape()[::-1])
        return out

    def get_contents(self):
        return [self.weight, self.bias]

    def get_ninput(self):
        return self.weight.shape()[0]

    def backward(self, X, gY):
        gW = np.tensordot(X, gY, axes=(self.bpAxes, self.bpAxes))
        gB = np.sum(gY, axis=self.bpAxes)
        gX = np.tensordot(gY, self.weight.get().T, axes=((-1),(0)))
        return gX, gW, gB

    def backward_connect(self, X, gY):
        gW = tc.contract(X, gY, [(1, 1), (2, 2)])
        gB = tc.reduce_sum(gY, 1, 2)
        gX = tc.contract(gY, tc.transpose(self.weight))
        return gX, gW, gB

# Define layer that unfolds the states over time
class RecurrentStateUnfold(object): # same thing as sequential dense + tanh
    """Unfold the recurrent states."""
    def __init__(self, nbStates):
        """Initialse the shared parameters, the inital state and
        state update function."""
        self.state0 = eteq.Variable([nbStates], label="init_state")
        self.linear = Dense(nbStates, nbStates, 2)
        self.tanh = rcn.tanh()

    def connect(self, X):
        """Iteratively apply forward step to all states."""
        # State tensor
        nbStates = self.linear.get_ninput()
        states = [tc.best_extend(self.state0, [nbStates, 1, X.shape()[0]])]
        for i in range(X.shape()[1]):
            # Update the states iteratively
            Xslice = tc.slice(X, i, 1, 1)
            fwd_state = self.linear.connect(states[-1])
            states.append(self.tanh.connect(Xslice + fwd_state))
        return tc.concat(states[1:], 1)

    def get_contents(self):
        return [self.state0] + self.linear.get_contents() + self.tanh.get_contents()

    def backward(self, X, S, gY):
        # Initialise gradient of state outputs
        nbTimesteps = X.shape[1]
        gSk = np.zeros_like(gY[:,0,:])
        # Initialse gradient tensor for state inputs
        gZ = np.zeros_like(X)
        gWSum = np.zeros_like(self.linear.weight.get())  # Initialise weight gradients
        gBSum = np.zeros_like(self.linear.bias.get())  # Initialse bias gradients
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

    def backward_connect(self, X, S, gY):
        # Initialise gradient of state outputs
        nbTimesteps = X.shape()[1]
        # Initialse gradient tensor for state inputs
        gZ = [None] * nbTimesteps
        yslice = gY.shape()
        yslice[1] = 1
        gSk = eteq.scalar_constant(0, yslice)
        gWSum = eteq.scalar_constant(0, self.linear.weight.shape())  # Initialise weight gradients
        gBSum = eteq.scalar_constant(0, self.linear.bias.shape())  # Initialse bias gradients

        # Propagate the gradients iteratively
        for k in range(nbTimesteps-1, -1, -1):
            # Gradient at state output is gradient from previous state
            #  plus gradient from output
            gSk = gSk + tc.slice(gY, k, 1, 1)
            # Propgate the gradient back through one state

            gZ_tmp = (1.0 - tc.pow(tc.slice(S, k+1, 1, 1), 2)) * gSk
            gSk, gW, gB = self.linear.backward_connect(tc.slice(S, k, 1, 1), gZ_tmp)
            gZ[k] = gZ_tmp
            gWSum = gWSum + gW  # Update total weight gradient
            gBSum = gBSum + gB  # Update total bias gradient
        # Get gradient of initial state over all samples
        gS0 = tc.reduce_sum(gSk, 2)
        return tc.concat(gZ, 1), gWSum, gBSum, gS0

# Define the full network
class RnnBinaryAdder(object): # !new layer: horizontal model
    """RNN to perform binary addition of 2 numbers."""
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states):
        self.tensorInput = Dense(nb_of_inputs, nb_of_states, 3)
        # Recurrent layer
        self.rnnUnfold = RecurrentStateUnfold(nb_of_states) # !new layer: horizontal model
        # Linear output transform
        self.tensorOutput = Dense(nb_of_states, nb_of_outputs, 3)
        self.classifier = rcn.sigmoid()  # Classification output

    def connect(self, X):
        _, _, Y = self.full_connect(X)
        return Y

    def full_connect(self, X):
        # Linear input transformation]
        recIn = self.tensorInput.connect(X)
        # Forward propagate through time and return states
        states = self.rnnUnfold.connect(recIn)
        # Linear output transformation
        Y = self.classifier.connect(self.tensorOutput.connect(states))
        return recIn, states, Y

    def backward(self, X, Y, recIn, S, T):
        sequence_len = X.shape[1]
        gError = (Y - T) / (Y.shape[0] * Y.shape[1])

        gRecOut, gWout, gBout = self.tensorOutput.backward(
            S[:,1:sequence_len+1,:], gError)
        # Propagate gradient backwards through time
        gRnnIn, gWrec, gBrec, gS0 = self.rnnUnfold.backward(
            recIn, S, gRecOut)
        _, gWin, gBin = self.tensorInput.backward(X, gRnnIn)
        return gWout, gBout, gWrec, gBrec, gWin, gBin, gS0

    def backward_connect(self, X, gotResult, recIn, state0, S, expectResult):
        gError = (gotResult - expectResult) / np.prod(gotResult.shape())
        gRecOut, gWout, gBout = self.tensorOutput.backward_connect(S, gError)
        States = tc.concat(tc.best_extend(state0, [3, 1, X.shape()[0]]), S, 1)
        gRnnIn, gWrec, gBrec, gS0 = self.rnnUnfold.backward_connect(
            recIn, States, gRecOut)
        _, gWin, gBin = self.tensorInput.backward_connect(X, gRnnIn)
        return gWout, gBout, gWrec, gBrec, gWin, gBin, gS0

    def get_contents(self):
        return self.tensorInput.get_contents() +\
            self.rnnUnfold.get_contents() +\
            self.tensorOutput.get_contents() +\
            self.classifier.get_contents()\

class RnnTrainer(object):
    def __init__(self, rnn, Xshape, Yshape, learning_rate, momentum_term, lmbd, eps):
        self.rnn = rnn

        self.vars = [
            rnn.tensorOutput.weight,
            rnn.tensorOutput.bias,
            rnn.rnnUnfold.linear.weight,
            rnn.rnnUnfold.linear.bias,
            rnn.tensorInput.weight,
            rnn.tensorInput.bias,
            rnn.rnnUnfold.state0,
        ]
        self.momentums = [np.zeros(v.shape()) for v in self.vars]
        self.moving_avg_sqr = [np.zeros(v.shape()) for v in self.vars]

        self.params = (learning_rate, momentum_term, lmbd, eps)

        self.sess = eteq.Session()
        self.inputX = eteq.Variable(Xshape)
        self.expectY = eteq.Variable(Yshape)

        # recIn, S, Y = self.rnn.full_connect(self.inputX)
        # self.grads = self.rnn.backward_connect(self.inputX, Y,
        #     recIn, self.rnn.rnnUnfold.state0, S, self.expectY)

        Y = self.rnn.connect(self.inputX)
        self.error = loss(Y, self.expectY)
        self.grads =  [eteq.derive(self.error, var) for var in self.vars]

        self.sess.track(list(self.grads) + [self.error])
        self.sess.optimize("cfg/optimizations.rules")

    def train(self, X, T):
        self.inputX.assign(X)
        self.expectY.assign(T)

        # apply rms prop optimization
        # group 1
        learning_rate, momentum_term, lmbd, eps = self.params
        momentum_tmp = [mom * momentum_term for mom in self.momentums]
        for var, mom_tmp in zip(self.vars, momentum_tmp):
            var.assign(np.array(var.get() + mom_tmp))

        self.sess.update_target(self.grads)
        grads = [grad.get() for grad in self.grads]

        # group 2
        for i in range(len(self.vars)):
            self.moving_avg_sqr[i] = lmbd * self.moving_avg_sqr[i] + (1-lmbd) * grads[i]**2

        # group 3
        pgrad_norms = [(learning_rate * grad) / np.sqrt(mvsqr) + eps
            for grad, mvsqr in zip(grads, self.moving_avg_sqr)]

        for i in range(len(self.vars)):
            self.momentums[i] = momentum_tmp[i] - pgrad_norms[i]

        for var, pgrad_norm in zip(self.vars, pgrad_norms):
            var.assign(np.array(var.get() - pgrad_norm))

    def get_error(self):
        self.sess.update_target([self.error])
        return self.error.get()

# Create dataset
nb_train = 2000  # Number of training samples
# Addition of 2 n-bit numbers can result in a n+1 bit number
sequence_len = 7  # Length of the binary sequence

# Create training samples
X_train, T_train = create_dataset(nb_train, sequence_len)
print(f'X_train tensor shape: {X_train.shape}')
print(f'T_train tensor shape: {T_train.shape}')

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
# Set the initial parameters
# Number of parameters in the network
nb_test = 5

sess = eteq.Session()

train_Xs = eteq.Variable([mb_size, sequence_len, 2])
train_Ys = eteq.Variable([mb_size, sequence_len, 1])

trainer = RnnTrainer(RNN, train_Xs.shape(), train_Ys.shape(),
    learning_rate, momentum_term, lmbd, eps)

test_Xs = eteq.Variable([nb_test, sequence_len, 2])
test_Ys = RNN.connect(test_Xs)
test_Ys = tc.round(test_Ys)
sess.track([test_Ys])

# sess.optimize("cfg/optimizations.rules")

trainer.inputX.assign(X_train[0:mb_size,:,:])
trainer.expectY.assign(T_train[0:mb_size,:,:])
ls_of_loss = [trainer.get_error()]
# Iterate over some iterations
start = time.time()
for i in range(5):
    # Iterate over all the minibatches
    for mb in range(nb_train // mb_size):
        X_mb = X_train[mb:mb+mb_size,:,:]  # Input minibatch
        T_mb = T_train[mb:mb+mb_size,:,:]  # Target minibatch

        train_Xs.assign(X_mb)
        train_Ys.assign(T_mb)

        # trainer.train(sess)
        trainer.train(X_mb, T_mb)

        # Add loss to list to plot
        ls_of_loss.append(trainer.get_error())
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
Xtest, Ttest = create_dataset(nb_test, sequence_len)
test_Xs.assign(Xtest)
# Push test data through network
sess.update_target([test_Ys])
Y = test_Ys.get()

# Print out all test examples
for i in range(Xtest.shape[0]):
    printSample(Xtest[i,:,0], Xtest[i,:,1], Ttest[i,:,:], Y[i,:,:])
    print('')
