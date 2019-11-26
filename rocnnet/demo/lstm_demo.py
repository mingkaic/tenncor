# source: https://github.com/Manik9/LSTMs

import numpy as np

import random
import math

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
    return eteq.variable(np.array(np.random.uniform(-0.1, 0.1, slist)), label)

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct

        # self.gate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
        #     weight_init=weight_init, bias_init=weight_init)
        # self.ingate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
        #     weight_init=weight_init, bias_init=weight_init)
        # self.forget = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
        #     weight_init=weight_init, bias_init=weight_init)
        # self.outgate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
        #     weight_init=weight_init, bias_init=weight_init)

        # weight matrices
        self.wg = rand_arr([mem_cell_ct, concat_len])
        self.wi = rand_arr([mem_cell_ct, concat_len])
        self.wf = rand_arr([mem_cell_ct, concat_len])
        self.wo = rand_arr([mem_cell_ct, concat_len])
        # bias terms
        self.bg = rand_arr([mem_cell_ct])
        self.bi = rand_arr([mem_cell_ct])
        self.bf = rand_arr([mem_cell_ct])
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

class LstmNode:
    def __init__(self, lstm_param):
        # store reference to parameters and to activations
        mem_cell_ct = lstm_param.mem_cell_ct
        self.old_state = np.zeros(mem_cell_ct)
        self.old_hidden = np.zeros(mem_cell_ct)

        self.state = eteq.variable([mem_cell_ct], 0, 'state')
        self.hidden = eteq.variable([mem_cell_ct], 0, 'hidden')

        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    def connect(self, x, s_prev, h_prev):
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # xc_placeholder = tc.variable([len(x), 2])
        # gate = tc.tanh(self.param.gate.connect(xc_placeholder))
        # inp = tc.sigmoid(self.param.ingate.connect(xc_placeholder))
        # forget = tc.sigmoid(self.param.forget.connect(xc_placeholder))
        # out = tc.sigmoid(self.param.outgate.connect(xc_placeholder))
        # next_state = gate * inp + s_prev * forget
        # next_hidden = self.state + out


        # concatenate x(t) and h(t-1)
        xc = np.hstack((x,  h_prev))
        self.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.old_state = self.g * self.i + s_prev * self.f
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

class LstmNetwork():
    def __init__(self, lstm_param, nstates):
        self.lstm_param = lstm_param
        self.lstm_node_list = [LstmNode(self.lstm_param) for _ in range(nstates)]

    def train(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence
        with corresponding loss layer.
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.lstm_node_list)

        err = np.zeros_like(y_list[-1]) # var
        diff_h = np.zeros(self.lstm_param.mem_cell_ct) # var
        diff_s = np.zeros(self.lstm_param.mem_cell_ct) # var

        for node, y in list(zip(self.lstm_node_list, y_list))[::-1]:
            err += loss_layer.loss(node.old_hidden, y)
            diff_h += loss_layer.bottom_diff(node.old_hidden, y)
            diff_h, diff_s = node.grad(diff_h, diff_s)

        return err

    def assign(self, xs):
        assert(len(xs) == len(self.lstm_node_list))

        s_prev = np.zeros_like(self.lstm_node_list[0].old_state)
        h_prev = np.zeros_like(self.lstm_node_list[0].old_hidden)
        for node, x in zip(self.lstm_node_list, xs):
            s_prev, h_prev = node.connect(x, s_prev, h_prev)

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

def main():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    y_list = [-0.5, 0.2, 0.1, -0.5]
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param, len(y_list))
    input_val_arr = [np.random.random(x_dim) for _ in y_list]

    for cur_iter in range(100):
        print("iter", "%2s" % str(cur_iter), end=": ")
        lstm_net.assign(input_val_arr)

        print("y_pred = [" +
              ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].old_hidden[0] for ind in range(len(y_list))]) +
              "]", end=", ")

        loss = lstm_net.train(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)
        lstm_param.apply_diff(lr=0.1)

if __name__ == "__main__":
    main()
