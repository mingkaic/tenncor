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
    return -(T * tc.log(Y) + (1-T) * tc.log(1-Y))

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

# Define layer that unfolds the states over time
class RecurrentStateUnfold(object): # same thing as sequential dense + tanh
    """Unfold the recurrent states."""
    def __init__(self, nbStates):
        """Initialse the shared parameters, the inital state and
        state update function."""
        self.state0 = eteq.Variable([nbStates], label="init_state")
        self.linear = Dense(nbStates, nbStates, 2)
        # self.linear = rcn.Dense(nbStates, nbStates)
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
        out = states[1]
        for state in states[2:]:
            out = tc.concat(out, state, 1)
        return out

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

# Define the full network
class RnnBinaryAdder(object): # !new layer: horizontal model
    """RNN to perform binary addition of 2 numbers."""
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states):
        self.tensorInput = Dense(nb_of_inputs, nb_of_states, 3)
        # self.tensorInput = rcn.Dense(nb_of_states, nb_of_inputs)
        # Recurrent layer
        self.rnnUnfold = RecurrentStateUnfold(nb_of_states) # !new layer: horizontal model
        # Linear output transform
        self.tensorOutput = Dense(nb_of_states, nb_of_outputs, 3)
        # self.tensorOutput = rcn.Dense(nb_of_outputs, nb_of_states)
        self.classifier = rcn.sigmoid()  # Classification output

    def connect(self, X):
        _, _, _, Y = self.full_connect(X)
        return Y

    def full_connect(self, X):
        # Linear input transformation]
        recIn = self.tensorInput.connect(X)
        # Forward propagate through time and return states
        states = self.rnnUnfold.connect(recIn)
        # Linear output transformation
        Z = self.tensorOutput.connect(states)
        Y = self.classifier.connect(Z)
        return recIn, states, Z, Y

    def backward(self, X, Y, recIn, S, T):
        sequence_len = X.shape[1]
        gZ = (Y - T) / (Y.shape[0] * Y.shape[1])
        gRecOut, gWout, gBout = self.tensorOutput.backward(
            S[:,1:sequence_len+1,:], gZ)
        # Propagate gradient backwards through time
        gRnnIn, gWrec, gBrec, gS0 = self.rnnUnfold.backward(
            recIn, S, gRecOut)
        _, gWin, gBin = self.tensorInput.backward(X, gRnnIn)
        return gWout, gBout, gWrec, gBrec, gWin, gBin, gS0

    def get_contents(self):
        return self.tensorInput.get_contents() +\
            self.rnnUnfold.get_contents() +\
            self.tensorOutput.get_contents() +\
            self.classifier.get_contents()\

class RnnTrainer(object):
    def __init__(self, rnn, sess, Xs, Ys, learning_rate, momentum_term, lmbd):
        self.err = loss(rnn.connect(Xs), Ys)

        iW, ib,\
        s0, ufW, ufb, _,\
        oW, ob, _ = tuple(rnn.get_contents())

        vargrad =  [
            (s0, eteq.derive(self.err, s0)),
            (iW, eteq.derive(self.err, iW)),
            (ib, eteq.derive(self.err, ib)),
            (ufW, eteq.derive(self.err, ufW)),
            (ufb, eteq.derive(self.err, ufb)),
            (oW, eteq.derive(self.err, oW)),
            (ob, eteq.derive(self.err, ob))]

        # apply rms prop optimization
        self.assignments = ([], [])
        to_track = [self.err]
        for target, grad in vargrad:
            momentum = eteq.Variable(target.shape(), scalar=0)
            movingAverage = eteq.Variable(target.shape(), scalar=0)

            # group 1
            momentum_incr = momentum * momentum_term

            target_incr = target + momentum_incr
            self.assignments[0].append((target, target_incr))

            # group 2
            nextMovingAverage = lmbd * movingAverage + (1-lmbd) * tc.pow(grad, 2)
            pGradNorm = (learning_rate * grad) / tc.sqrt(nextMovingAverage) + eps
            next_momentum = momentum_incr - pGradNorm
            nextTarget = target - pGradNorm

            self.assignments[1].append((movingAverage, nextMovingAverage))
            self.assignments[1].append((momentum, next_momentum))
            self.assignments[1].append((target, nextTarget))
            to_track += [target_incr, nextMovingAverage, next_momentum, nextTarget]

        sess.track(to_track)

    def train(self, sess):
        for group in self.assignments:
            sources = [source for _, source in group]
            sess.update_target(sources)

            for target, source in group:
                target.assign(source.get())

class RnnTrainer2(object):
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
        self.movingAvgSqr = [np.zeros(v.shape()) for v in self.vars]

        self.params = (learning_rate, momentum_term, lmbd, eps)

        self.sess = eteq.Session()
        self.inputX = eteq.Variable(Xshape)
        self.expectY = eteq.Variable(Yshape)
        self.recIn, self.S, _, self.Y = self.rnn.full_connect(self.inputX)
        self.sess.track([self.recIn, self.S, self.Y])

    def train(self, X, T):
        learning_rate, momentum_term, lmbd, eps = self.params

        momentum_tmp = [v * momentum_term for v in self.momentums]
        for var, mom_tmp in zip(self.vars, momentum_tmp):
            var.assign(np.array(var.get() + mom_tmp))

        self.inputX.assign(X)
        self.sess.update_target([self.recIn, self.S, self.Y])
        States = np.zeros([100, 8, 3])
        States[:,0,:] = self.rnn.rnnUnfold.state0.get()
        States[:,1:,:] = self.S.get()
        grads = self.rnn.backward(X, self.Y.get(), self.recIn.get(), States, T)

        # update moving average
        for i in range(len(self.vars)):
            self.movingAvgSqr[i] = lmbd * self.movingAvgSqr[i] + (1-lmbd) * grads[i]**2

        pGradNorms = [(learning_rate * grad) / np.sqrt(mvsqr) + eps
            for grad, mvsqr in zip(grads, self.movingAvgSqr)]

        # update momentums
        for i in range(len(self.vars)):
            self.momentums[i] = momentum_tmp[i] - pGradNorms[i]

        for var, pgrad_norm in zip(self.vars, pGradNorms):
            var.assign(np.array(var.get() - pgrad_norm))

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

train_Xs.assign(X_train[0:mb_size,:,:])
train_Ys.assign(T_train[0:mb_size,:,:])

trainer_bad = RnnTrainer(RNN, sess, train_Xs, train_Ys,
    learning_rate, momentum_term, lmbd)
trainer = RnnTrainer2(RNN, train_Xs.shape(), train_Ys.shape(),
    learning_rate, momentum_term, lmbd, eps)

test_Xs = eteq.Variable([nb_test, sequence_len, 2])
test_Ys = RNN.connect(test_Xs)
test_Ys = tc.round(test_Ys)
sess.track([test_Ys])

sess.optimize("cfg/optimizations.rules")

sess.update_target([trainer_bad.err])
ls_of_loss = [trainer_bad.err.get()]
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
        sess.update_target([trainer_bad.err])
        ls_of_loss.append(trainer_bad.err.get())
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
