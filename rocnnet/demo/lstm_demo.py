# logic source: https://github.com/Manik9/LSTMs

import time

import numpy as np

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

def loss(pred, label):
    return tc.pow(pred - label, 2)

def lstm_loss(label, predictions):
    return loss(tc.transpose(tc.slice(predictions, 0, 1, 0)), label)

def main():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    sess = eteq.Session()

    lstm_net = rcn.LSTM(mem_cell_ct, x_dim,
        weight_init=rcn.unif_xavier_init(1),
        bias_init=rcn.unif_xavier_init(1))

    test_inputs = eteq.variable(np.array(input_val_arr), 'test_input')
    test_outputs = eteq.variable(np.array(y_list), 'test_outputs')
    hiddens = tc.slice(lstm_net.connect(test_inputs), 0, 1, 0)
    err = tc.reduce_sum(loss(tc.transpose(tc.slice(hiddens, 0, 1, 0)), test_outputs))
    sess.track([hiddens, err])

    trainer = rcn.sgd_train(lstm_net, sess, test_inputs, test_outputs,
        rcn.get_sgd(learning_rate=0.1),
        errfunc=lstm_loss)

    # eteq.optimize(sess, eteq.parse_optrules("cfg/optimizations.rules"))

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
