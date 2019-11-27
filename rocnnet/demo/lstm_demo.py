# source: https://github.com/Manik9/LSTMs

import numpy as np

import random
import math
import time

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values):
    return values*(1-values)

def tanh_derivative(values):
    return 1. - values ** 2

# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(shape):
    return np.random.uniform(-0.1, 0.1, shape)

def weight_init(shape, label):
    slist = shape.as_list()
    a = np.sqrt(6.0 / np.sum(slist))
    return eteq.variable(np.array(np.random.uniform(-a, a, slist)), label)

def old_loss(pred, label):
    return (pred - label) ** 2

def old_loss_grad(pred, label):
    diff = np.zeros_like(pred)
    diff[0] = 2 * (pred[0] - label)
    return diff

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct

        self.wg = rand_arr([mem_cell_ct, concat_len])
        self.bg = rand_arr([mem_cell_ct])
        self.wi = rand_arr([mem_cell_ct, concat_len])
        self.bi = rand_arr([mem_cell_ct])
        self.wf = rand_arr([mem_cell_ct, concat_len])
        self.bf = rand_arr([mem_cell_ct])
        self.wo = rand_arr([mem_cell_ct, concat_len])
        self.bo = rand_arr([mem_cell_ct])
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def apply_diff(self, lr = 1):
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)

class OldLstmNode:
    def __init__(self, lstm_param):
        # store reference to parameters and to activations
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def old_connect(self, x, prev_state, prev_hidden):
        # save data for use in backprop
        self.s_prev = prev_state
        self.h_prev = prev_hidden

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  prev_hidden))
        self.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.old_state = self.g * self.i + prev_state * self.f
        self.old_hidden = self.old_state * self.o

        self.xc = xc

        return (self.old_state, self.old_hidden)

    def grad(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.o * top_diff_h + top_diff_s
        do = self.old_state * top_diff_h
        di = self.g * ds
        dg = self.i * ds
        df = self.s_prev * ds

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.i) * di
        df_input = sigmoid_derivative(self.f) * df
        do_input = sigmoid_derivative(self.o) * do
        dg_input = tanh_derivative(self.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        # compute bottom diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.bottom_diff_s = ds * self.f
        self.bottom_diff_h = dxc[self.param.x_dim:]

        return (self.bottom_diff_h, self.bottom_diff_s)

class OldLstmNetwork():
    def __init__(self, lstm_param, nstates):
        self.lstm_param = lstm_param
        self.lstm_node_list = [OldLstmNode(self.lstm_param) for _ in range(nstates)]

        mem_cell_ct = self.lstm_param.mem_cell_ct
        concat_len = self.lstm_param.x_dim + mem_cell_ct
        self.gate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=weight_init)
        self.ingate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=weight_init)
        self.forget = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=weight_init)
        self.outgate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=weight_init)

        self.nodes = [OldLstmNode((self.gate, self.forget, self.ingate, self.outgate))
            for _ in range(nstates)]

    def old_train(self, y_list, lr):
        err = np.zeros_like(y_list[-1]) # var
        diff_h = np.zeros(self.lstm_param.mem_cell_ct) # var
        diff_s = np.zeros(self.lstm_param.mem_cell_ct) # var

        for node, y in list(zip(self.lstm_node_list, y_list))[::-1]:
            err += old_loss(node.old_hidden[0], y)
            diff_h += old_loss_grad(node.old_hidden, y)
            diff_h, diff_s = node.grad(diff_h, diff_s)
        self.lstm_param.apply_diff(lr)

        return err

    def old_connect(self, xs):
        states = []
        state = np.zeros(self.lstm_param.mem_cell_ct)
        hidden = np.zeros(self.lstm_param.mem_cell_ct)
        for node, x in zip(self.lstm_node_list, xs):
            state, hidden = node.old_connect(x, state, hidden)
            states.append((state, hidden))
        return states

def loss(pred, label):
    return tc.pow(pred - label, 2)

def update_grad(grads, lr):
    group = []
    for var, grad in grads:
        group.append((var, var - lr * grad))

    return [group]

class LstmNetwork():
    def __init__(self, mem_cell_ct, x_dim, nstates):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim

        concat_len = x_dim + mem_cell_ct
        self.gate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=weight_init)
        self.ingate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=weight_init)
        self.forget = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=weight_init)
        self.outgate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=weight_init)

    def get_contents(self):
        gate_vars = self.gate.get_contents()
        forgot_vars = self.forget.get_contents()
        ingate_vars = self.ingate.get_contents()
        outgate_vars = self.outgate.get_contents()
        return [
            gate_vars[0], gate_vars[1],
            forgot_vars[0], forgot_vars[1],
            ingate_vars[0], ingate_vars[1],
            outgate_vars[0], outgate_vars[1],
        ]

    def cell_connect(self, x, prev_state, prev_hidden):
        xc = tc.concat(x, prev_hidden, 0)

        gate = tc.tanh(self.gate.connect(xc))
        inp = tc.sigmoid(self.ingate.connect(xc))
        forget = tc.sigmoid(self.forget.connect(xc))
        out = tc.sigmoid(self.outgate.connect(xc))
        state = gate * inp + prev_state * forget
        hidden = state * out

        return (state, hidden)

    def connect(self, xinput):
        inshape = xinput.shape()
        nseq = inshape[0]
        states = []
        state = eteq.scalar_constant(0, [self.mem_cell_ct])
        hidden = eteq.scalar_constant(0, [self.mem_cell_ct])
        for i in range(nseq):
            state, hidden = self.cell_connect(
                tc.slice(xinput, i, 1, 1), state, hidden)
            states.append(hidden)
        return states

def LstmTrainer(model, sess, y_list, states, lr):
    hiddens = tc.concat(states, 1)
    err = loss(tc.transpose(tc.slice(hiddens, 0, 1, 0)), y_list)
    vars = model.get_contents()
    grads = [(var, eteq.derive(err, var)) for var in vars]

    updates = update_grad(grads, lr)
    changes = []
    to_track = []
    for group in updates:
        cgroup = [change for _, change in group]
        to_track += cgroup
        changes.append(cgroup)
    loss_var = tc.reduce_sum(err)
    sess.track(to_track + [loss_var])

    def update():
        for group, cgroup in zip(updates, changes):
            sess.update_target(cgroup)
            for var, diff in group:
                var.assign(diff.get())

    return update, loss_var

def main():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    y_list = [-0.5, 0.2, 0.1, -0.5]
    lstm_net = LstmNetwork(mem_cell_ct, x_dim, len(y_list))
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    test_inputs = eteq.variable(np.array(input_val_arr), 'test_input')
    test_outputs = eteq.variable(np.array(y_list), 'test_outputs')
    states = lstm_net.connect(test_inputs)
    sess = eteq.Session()
    trainer, loss_var = LstmTrainer(lstm_net, sess, test_outputs, states, lr=0.1)
    eteq.optimize(sess, eteq.parse_optrules("cfg/optimizations.rules"))

    start = time.time()
    for cur_iter in range(100):
        print("iter", "%2s" % str(cur_iter), end=": ")
        trainer()

        print("y_pred = [" +
              ", ".join(["% 2.5f" % hidden.get()[0] for hidden in states]) +
              "]", end=", ")

        sess.update_target([loss_var])
        print("loss:", "%.3e" % loss_var.get())
    print('training time: {} seconds'.format(time.time() - start))
    print()

    lstm_param = LstmParam(mem_cell_ct, x_dim)
    oldlstm_net = OldLstmNetwork(lstm_param, len(y_list))
    start = time.time()
    for cur_iter in range(100):
        print("iter", "%2s" % str(cur_iter), end=": ")
        old_states = oldlstm_net.old_connect(input_val_arr)

        print("y_pred = [" +
              ", ".join(["% 2.5f" % hidden[0] for _, hidden in old_states]) +
              "]", end=", ")

        loss = oldlstm_net.old_train(y_list, lr=0.1)
        print("loss:", "%.3e" % loss)
    print('training time 2: {} seconds'.format(time.time() - start))

if __name__ == "__main__":
    main()
