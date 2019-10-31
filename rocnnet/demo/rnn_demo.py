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

def loss(T, Y):
    return tc.reduce_mean(cross_entropy_loss(Y, T))

# Show an example input and target
def print_sample(x1, x2, t, y=None):
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

def weight_init(shape, label):
    slist = shape.as_list()
    a = np.sqrt(6.0 / np.sum(slist))
    return eteq.variable(np.array(np.random.uniform(-a, a, slist)), label)

def make_rms_prop(learning_rate, momentum_term, lmbd, eps):
    def rms_prop(grads):
        targets = [target for target, _ in grads]
        gs = [grad for _, grad in grads]
        momentum = [eteq.scalar_variable(0, target.shape()) for target in targets]
        mvavg_sqr = [eteq.scalar_variable(0, target.shape()) for target in targets]

        # group 1
        momentum_tmp = [mom * momentum_term for mom in momentum]
        group1 = [rcn.VarAssign(target, target + source) for target, source in zip(targets, momentum_tmp)]

        # group 2
        group2 = [rcn.VarAssign(target, lmbd * target + (1-lmbd) * tc.pow(grad, 2)) for target, grad in zip(mvavg_sqr, gs)]

        # group 3
        pgrad_norm_nodes = [(learning_rate * grad) / tc.sqrt(mvsqr) + eps
            for grad, mvsqr in zip(gs, mvavg_sqr)]

        group3 = [rcn.VarAssign(target, momentum_tmp_node - pgrad_norm_node)
            for target, momentum_tmp_node, pgrad_norm_node in zip(momentum, momentum_tmp, pgrad_norm_nodes)]

        group3 += [rcn.VarAssign(target, target - pgrad_norm) for target, pgrad_norm in zip(targets, pgrad_norm_nodes)]

        return [group1, group2, group3]

    return rms_prop

# def rnn_train(rnn, sess, train_input, train_exout,
#     learning_rate, momentum_term, lmbd, eps):

#     winp, binp, _, w, b, _, _, state0, wout, bout, _, _ = tuple(rnn.get_contents())

#     targets = [winp, binp, w, b, state0, wout, bout]
#     momentums = [np.zeros(v.shape()) for v in targets]
#     moving_avg_sqr = [np.zeros(v.shape()) for v in targets]

#     train_out = rnn.connect(train_input)
#     error = loss(train_exout, train_out)
#     grads =  [eteq.derive(error, var) for var in targets]

#     sess.track(grads)

#     # apply rms prop optimization
#     def train():
#         # group 1
#         momentum_tmp = [mom * momentum_term for mom in momentums]
#         for var, mom_tmp in zip(targets, momentum_tmp):
#             var.assign(np.array(var.get() + mom_tmp))

#         sess.update_target(grads)
#         res_grads = [grad.get() for grad in grads]

#         # group 2
#         for i in range(len(targets)):
#             moving_avg_sqr[i] = lmbd * moving_avg_sqr[i] + (1-lmbd) * res_grads[i]**2

#         # group 3
#         pgrad_norms = [(learning_rate * grad) / np.sqrt(mvsqr) + eps
#             for grad, mvsqr in zip(res_grads, moving_avg_sqr)]

#         for i in range(len(targets)):
#             momentums[i] = momentum_tmp[i] - pgrad_norms[i]

#         for var, pgrad_norm in zip(targets, pgrad_norms):
#             var.assign(np.array(var.get() - pgrad_norm))

#     return train

# Create dataset
nb_train = 2000  # Number of training samples
# Addition of 2 n-bit numbers can result in a n+1 bit number
sequence_len = 7  # Length of the binary sequence

# Create training samples
xtrain, expect_train = create_dataset(nb_train, sequence_len)
print(f'xtrain tensor shape: {xtrain.shape}')
print(f'expect_train tensor shape: {expect_train.shape}')

# keep this here to get nice weights
rcn.Dense(2, eteq.Shape([3]), weight_init)
rcn.Dense(3, eteq.Shape([3]), weight_init)
rcn.Dense(3, eteq.Shape([1]), weight_init)

# Set hyper-parameters
lmbd = 0.5  # Rmsprop lambda
learning_rate = 0.05  # Learning rate
momentum_term = 0.80  # Momentum term
eps = 1e-6  # Numerical stability term to prevent division by zero
mb_size = 100  # Size of the minibatches (number of samples)

# Create the network
nunits = 3  # Number of states in the recurrent layer
ninput = 2
noutput = 1

model = rcn.SequentialModel("demo")
model.add(
    rcn.Dense(nunits, eteq.Shape([ninput]),
    weight_init, label="input"))
model.add(rcn.Recur(nunits, rcn.tanh(),
    weight_init=weight_init,
    bias_init=rcn.zero_init(), label="unfold"))
model.add(rcn.Dense(noutput, eteq.Shape([nunits]),
    weight_init, label="output"))
model.add(rcn.sigmoid(label="classifier"))

# Number of parameters in the network
nb_test = 5

sess = eteq.Session()

train_input = eteq.Variable([mb_size, sequence_len, ninput])
train_exout = eteq.Variable([mb_size, sequence_len, noutput])

got = model.connect(train_input)
error = loss(got, train_exout)
sess.track([error])

train = rcn.sgd_train(model, sess, train_input, train_exout,
    make_rms_prop(learning_rate, momentum_term, lmbd, eps),
    errfunc=loss)
# train = rnn_train(model, sess, train_input, train_exout,
#     learning_rate, momentum_term, lmbd, eps)

test_input = eteq.Variable([nb_test, sequence_len, ninput])
test_output = tc.round(model.connect(test_input))
sess.track([test_output])

sess.optimize("cfg/optimizations.rules")

train_input.assign(xtrain[0:mb_size,:,:])
train_exout.assign(expect_train[0:mb_size,:,:])
sess.update_target([error])
ls_of_loss = [error.get()]
# Iterate over some iterations
start = time.time()
for i in range(5):
    # Iterate over all the minibatches
    for mb in range(nb_train // mb_size):
        xbatch = xtrain[mb:mb+mb_size,:,:]
        tbatch = expect_train[mb:mb+mb_size,:,:]

        train_input.assign(xbatch)
        train_exout.assign(tbatch)

        train()

        # Add loss to list to plot
        sess.update_target([error])
        ls_of_loss.append(error.get())
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

xtest, expect_test = create_dataset(nb_test, sequence_len)

test_input.assign(xtest)
sess.update_target([test_output])
got_test = test_output.get()

for i in range(xtest.shape[0]):
    print_sample(xtest[i,:,0], xtest[i,:,1], expect_test[i,:,:], got_test[i,:,:])
    print('')
