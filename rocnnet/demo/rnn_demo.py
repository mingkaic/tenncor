# source: http://peterroelants.github.io/posts/rnn-implementation-part01/
# Imports
import sys

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

import dbg.stream_dbg as dbg

import numpy as np  # Matrix and vector computation package

# Set the seed for reproducability
np.random.seed(seed=1)
#

# equivalent to [xweight, sweight] @ concat([[inputs...], [states...]])
def connect(xweight, sweight, xs, activation = None):
    assert len(xs) > 0
    shape = xs[0].shape()
    states = [eteq.scalar_constant(0, shape)]
    for x in xs:
        step = x * tc.extend(xweight, 0, shape[::-1]) +\
            states[-1] * tc.extend(sweight, 0, shape[::-1])
        if activation is not None:
            step = activation(step)
        states.append(step)
    return states


np.random.seed(seed=1)

nb_of_samples = 20
sequence_len = 10
# Create the sequences
X = np.zeros((nb_of_samples, sequence_len))
for row_idx in range(nb_of_samples):
    X[row_idx,:] = np.around(np.random.rand(sequence_len)).astype(int)
# Create the targets for each sequence
t = np.sum(X, axis=1)

# Set hyperparameters
eta_p = 1.2
eta_n = 0.5

# Set initial parameters
W = [-1.5, 2]  # [wx, wRec]
W_delta = [0.001, 0.001]  # Update values (Delta) for W


xweight = eteq.Variable([1], label='xweight', scalar=W[0])
sweight = eteq.Variable([1], label='sweight', scalar=W[1])
xs = [eteq.Variable([nb_of_samples], label='x_{}'.format(i)) for i in range(sequence_len)]
expectFinal = eteq.Variable([nb_of_samples], label='expect')

for i, x in enumerate(xs):
    x.assign(X[:,i])
expectFinal.assign(t)


states = connect(xweight, sweight, xs)
lossed = tc.reduce_mean(tc.pow(states[-1] - expectFinal, 2))
# derive lossed wrt to (xweight, sweight)
grad_xweight = eteq.derive(lossed, xweight)
grad_sweight = eteq.derive(lossed, sweight)

# update_rprop
prev_xsign = eteq.Variable([1], label='xsign')
prev_ssign = eteq.Variable([1], label='ssign')
xsign = tc.sign(grad_xweight)
ssign = tc.sign(grad_sweight)
next_xweight = xweight - W_delta[0] * xsign * \
    tc.if_then_else(xsign == prev_xsign,
    eteq.scalar_constant(eta_p, []),
    eteq.scalar_constant(eta_n, []))
next_sweight = sweight - W_delta[1] * ssign * \
    tc.if_then_else(ssign == prev_ssign,
    eteq.scalar_constant(eta_p, []),
    eteq.scalar_constant(eta_n, []))

def train(sess):
    sess.update_target([next_xweight, next_sweight])
    print(grad_xweight.get())
    print(xsign.get())
    print(next_xweight.get())
    print(grad_sweight.get())
    print(ssign.get())
    print(next_sweight.get())
    xweight.assign(next_xweight.get())
    sweight.assign(next_sweight.get())
    prev_xsign.assign(xsign.get())
    prev_ssign.assign(ssign.get())
    raise Exception('ok')

sess = eteq.Session()
sess.track([next_xweight, next_sweight])
sess.optimize("cfg/optimizations.rules")

abbrevs = dict()
for i, state in enumerate(states):
    abbrevs[state.as_tens()] = 'state_{}'.format(i)
dbg.multigraph_to_csvfile(
    [grad_xweight.as_tens(), grad_sweight.as_tens()],
    '/tmp/rnn_graph.csv', abbrevs=abbrevs)

for i in range(500):
    train(sess)


xin = np.asmatrix([[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1]])
test_input = eteq.Variable([12], label='x')
test_output = connect(xweight, sweight, [test_input])[0]
test_input.assign(xin)
sess.track([test_output])
sess.update_target([test_output])

sum_test_input = xin.sum()
print('Target output: {} vs Model output: {}'.format(
    sum_test_input, test_output.get()))
