# source: https://peterroelants.github.io/posts/rnn-implementation-part02/
import sys
import functools

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

import numpy as np  # Matrix and vector computation package
import matplotlib
import matplotlib.pyplot as plt  # Plotting library

# Set the seed for reproducability
np.random.seed(seed=1)

# Create dataset
nb_train = 2000  # Number of training samples
# Addition of 2 n-bit numbers can result in a n+1 bit number
sequence_len = 7  # Length of the binary sequence

def loss(Ys, Ts):
    concat = lambda left, right: tc.concat(left, right, 0)
    Y = functools.reduce(concat, Ys)
    T = functools.reduce(concat, Ts)
    return -tc.reduce_mean(T * tc.log(Y) + (1-T) * tc.log(1-Y))

# Create a list of minibatch losses to be plotted
def batch_assign(vars, data):
    for var, d in zip(vars, [data[:,i,:] for i in range(data.shape[1])]):
        var.assign(d)

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

# Define layer that unfolds the states over time
class RecurrentStateUnfold(object): # same thing as sequential dense + tanh
    """Unfold the recurrent states."""
    def __init__(self, nbStates):
        """Initialse the shared parameters, the inital state and
        state update function."""
        self.state0 = eteq.Variable([nbStates], label="init_state")
        weight = eteq.variable(np.array([
            [ 0.92415643, -0.63100879,  0.01979676],
            [-0.31242379,  0.53945069,  0.60573297],
            [-0.15088018, -0.591755,   -0.86582108]]), 'weight')
        bias = eteq.variable(np.array([0., 0., 0.]), 'bias')
        self.linear = rcn.create_dense(weight, bias)
        # self.linear = rcn.Dense(nbStates, nbStates)
        self.tanh = rcn.tanh()

    def connect(self, Xs):
        """Iteratively apply forward step to all states."""
        # State tensor
        states = [self.state0]
        for X in Xs:
            # Update the states iteratively
            states.append(self.tanh.connect(X + \
                tc.best_extend(self.linear.connect(states[-1]),
                    X.shape()[::-1])))
        return states

    def get_contents(self):
        return [self.state0] + self.linear.get_contents() + self.tanh.get_contents()

# Define the full network
class RnnBinaryAdder(object): # !new layer: horizontal model
    """RNN to perform binary addition of 2 numbers."""
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states):
        weightInput = eteq.variable(np.array([
            [ 0.16865001,  0.82243552,  0.23785496],
            [-0.54408659, -0.44665696,  0.07212971]]), 'weight')
        biasInput = eteq.variable(np.array([0., 0., 0.]), 'bias')
        self.tensorInput = rcn.create_dense(weightInput, biasInput)

        weightOutput = eteq.variable(np.array([
            [-0.73815799],
            [-0.55750178],
            [ 0.24198244]]), 'weight')
        biasOutput = eteq.variable(np.array([0.]), 'bias')
        self.tensorOutput = rcn.create_dense(weightOutput, biasOutput)

        # self.tensorInput = rcn.Dense(nb_of_states, nb_of_inputs)
        # Recurrent layer
        self.rnnUnfold = RecurrentStateUnfold(nb_of_states) # !new layer: horizontal model
        # Linear output transform
        # self.tensorOutput = rcn.Dense(nb_of_outputs, nb_of_states)
        self.classifier = rcn.sigmoid()  # Classification output

    def connect(self, Xs):
        # Linear input transformation
        recIn = [self.tensorInput.connect(X) for X in Xs]
        # Forward propagate through time and return states
        states = self.rnnUnfold.connect(recIn)
        # Linear output transformation
        return [self.classifier.connect(self.tensorOutput.connect(state))
            for state in states[1:]]

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

        initAssignments =  [
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
        for target, source in initAssignments:
            movingAverage = eteq.Variable(target.shape(), scalar=0)
            momentum = eteq.Variable(target.shape(), scalar=0)

            # group 1
            momentum_incr = momentum * momentum_term

            target_incr = target + momentum_incr
            self.assignments[0].append((target, target_incr))

            # group 2
            nextMovingAverage = lmbd * movingAverage + (1-lmbd) * tc.pow(source, 2)
            pGradNorm = ((learning_rate * source) / tc.sqrt(nextMovingAverage) + eps)
            next_momentum = momentum_incr - pGradNorm
            nextTarget = target + pGradNorm

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

train_Xs = [eteq.Variable([mb_size, 2]) for _ in range(sequence_len)]
train_Ys = [eteq.Variable([mb_size, 1]) for _ in range(sequence_len)]

batch_assign(train_Xs, X_train[0:100,:,:])
batch_assign(train_Ys, T_train[0:100,:,:])

trainer = RnnTrainer(RNN, sess, train_Xs, train_Ys,
    learning_rate, momentum_term, lmbd)

test_Xs = [eteq.Variable([nb_test, 2]) for _ in range(sequence_len)]
Yfs = RNN.connect(test_Xs)
sess.track(Yfs)

sess.optimize("cfg/optimizations.rules")

sess.update_target([trainer.err])
ls_of_loss = [trainer.err.get()]
print(ls_of_loss[0])
# Iterate over some iterations
for i in range(5):
    # Iterate over all the minibatches
    for mb in range(nb_train // mb_size):
        batch_assign(train_Xs, X_train[mb:mb+mb_size,:,:])
        batch_assign(train_Ys, T_train[mb:mb+mb_size,:,:])

        trainer.train(sess)

        # Add loss to list to plot
        sess.update_target([trainer.err])
        ls_of_loss.append(trainer.err.get())

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
# Push test data through network
sess.update_target(Yfs)
Y = np.array([tc.round(Yf).get() for Yf in Yfs])

# Print out all test examples
for i in range(Xtest.shape[0]):
    printSample(Xtest[i,:,0], Xtest[i,:,1], Ttest[i,:,:], Y[i,:,:])
    print('')
