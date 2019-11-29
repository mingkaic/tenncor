# logic source: https://github.com/Manik9/LSTMs
import sys
import time
import argparse

import numpy as np

import eteq.tenncor as tc
import eteq.eteq as eteq
import rocnnet.rocnnet as rcn

prog_description = 'Demo lstm'

def loss(pred, label):
    return tc.pow(pred - label, 2)

def lstm_loss(label, predictions):
    return loss(tc.transpose(tc.slice(predictions, 0, 1, 0)), label)

def str2bool(opt):
    optstr = opt.lower()
    if optstr in ('yes', 'true', 't', 'y', '1'):
        return True
    elif optstr in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    default_ts = time.time()

    parser = argparse.ArgumentParser(description=prog_description)
    parser.add_argument('--seed', dest='seed',
        type=str2bool, nargs='?', const=False, default=False,
        help='Whether to seed or not (default: True)')
    parser.add_argument('--seedval', dest='seedval', type=int, nargs='?', default=int(default_ts),
        help='Random seed value (default: <current time>)')
    parser.add_argument('--n_train', dest='n_train', type=int, nargs='?', default=100,
        help='Number of times of train (default: 100)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='models/lstm.pbx',
        help='Filename to load pretrained model (default: models/lstm.pbx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        eteq.seed(args.seedval)
        np.random.seed(args.seedval)
    else:
        np.random.seed(seed=0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    y_list = [-0.5, 0.2, 0.1, -0.5]
    input_val_arr = [np.random.random(x_dim) for _ in y_list]
    sess = eteq.Session()

    lstm = rcn.LSTM(mem_cell_ct, x_dim,
        weight_init=rcn.unif_xavier_init(1),
        bias_init=rcn.unif_xavier_init(1))
    model = rcn.SequentialModel("model")
    model.add(lstm)
    untrained_model = model.clone()
    try:
        print('loading ' + args.load)
        pretrained_model = rcn.load_file_seqmodel(args.load, "model")
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))
        pretrained_model = model.clone()

    test_inputs = eteq.variable(np.array(input_val_arr), 'test_input')
    test_outputs = eteq.variable(np.array(y_list), 'test_outputs')

    untrained = tc.slice(untrained_model.connect(test_inputs), 0, 1, 0)
    hiddens = tc.slice(model.connect(test_inputs), 0, 1, 0)
    pretrained = tc.slice(pretrained_model.connect(test_inputs), 0, 1, 0)

    err = tc.reduce_sum(loss(tc.transpose(tc.slice(hiddens, 0, 1, 0)), test_outputs))
    sess.track([untrained, hiddens, pretrained, err])

    trainer = rcn.sgd_train(model, sess, test_inputs, test_outputs,
        rcn.get_sgd(learning_rate=0.1),
        errfunc=lstm_loss)

    # eteq.optimize(sess, eteq.parse_optrules("cfg/optimizations.rules"))

    start = time.time()
    for cur_iter in range(args.n_train):
        print("iter", "%2s" % str(cur_iter), end=": ")
        trainer()

        print("y_pred = [" +
              ", ".join(["% 2.5f" % state for state in hiddens.get()]) +
              "]", end=", ")

        sess.update_target([err])
        print("loss:", "%.3e" % err.get())

    sess.update_target([untrained, hiddens, pretrained])
    print("untrained_y_pred = {}".format(untrained.get().flatten()))
    print("trained_y_pred = {}".format(hiddens.get().flatten()))
    print("pretrained_y_pred = {}".format(pretrained.get().flatten()))
    print('training time: {} seconds'.format(time.time() - start))

    try:
        print('saving')
        if model.save_file(args.save):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if __name__ == "__main__":
    main(sys.argv[1:])
