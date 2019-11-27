# logic source: https://github.com/Manik9/LSTMs

import numpy as np

import random
import time

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

def loss(pred, label):
    return tc.pow(pred - label, 2)

def lstm_loss(label, predictions):
    return loss(tc.transpose(tc.slice(predictions, 0, 1, 0)), label)

class LstmNetwork():
    def __init__(self, mem_cell_ct, x_dim, weight_init, bias_init):
        self.mem_cell_ct = mem_cell_ct

        concat_len = x_dim + mem_cell_ct
        self.gate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=bias_init)
        self.ingate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=bias_init)
        self.forget = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=bias_init)
        self.outgate = rcn.Dense(mem_cell_ct, eteq.Shape([concat_len]),
            weight_init=weight_init, bias_init=bias_init)

    def _cell_connect(self, x, prev_state, prev_hidden):
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
            state, hidden = self._cell_connect(
                tc.slice(xinput, i, 1, 1), state, hidden)
            states.append(hidden)
        return tc.concat(states, 1)

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

def update_grad(grads, lr):
    group = []
    for var, grad in grads:
        group.append((var, var - lr * grad))

    return [group]

def LstmTrainer(model, sess, test_in, y_list, lr, errfunc=lstm_loss):
    err = errfunc(y_list, model.connect(test_in))
    vars = model.get_contents()
    grads = [(var, eteq.derive(err, var)) for var in vars]

    updates = update_grad(grads, lr)
    changes = []
    to_track = []
    for group in updates:
        cgroup = [change for _, change in group]
        to_track += cgroup
        changes.append(cgroup)
    sess.track(to_track)

    def update():
        for group, cgroup in zip(updates, changes):
            sess.update_target(cgroup)
            for var, diff in group:
                var.assign(diff.get())

    return update

def main():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    sess = eteq.Session()

    lstm_net = LstmNetwork(mem_cell_ct, x_dim,
        weight_init=rcn.unif_xavier_init(1),
        bias_init=rcn.unif_xavier_init(1))

    test_inputs = eteq.variable(np.array(input_val_arr), 'test_input')
    test_outputs = eteq.variable(np.array(y_list), 'test_outputs')
    hiddens = tc.slice(lstm_net.connect(test_inputs), 0, 1, 0)
    err = tc.reduce_sum(loss(tc.transpose(tc.slice(hiddens, 0, 1, 0)), test_outputs))
    sess.track([hiddens, err])

    trainer = LstmTrainer(lstm_net, sess, test_inputs, test_outputs, lr=0.1, errfunc=lstm_loss)
    # trainer = rcn.sgd_train(lstm_net, sess, test_inputs, test_outputs,
    #     rcn.get_sgd(learning_rate=0.1),
    #     errfunc=lstm_loss)

    eteq.optimize(sess, eteq.parse_optrules("cfg/optimizations.rules"))

    start = time.time()
    for cur_iter in range(100):
        print("iter", "%2s" % str(cur_iter), end=": ")
        trainer()

        print("y_pred = [" +
              ", ".join(["% 2.5f" % state for state in hiddens.get()]) +
              "]", end=", ")

        sess.update_target([err])
        print("loss:", "%.3e" % err.get())
    print('training time: {} seconds'.format(time.time() - start))

if __name__ == "__main__":
    main()
