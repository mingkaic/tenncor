# logic source: https://github.com/Manik9/LSTMs
import sys
import time
import argparse

import numpy as np

import eteq.tenncor as tc
import eteq.eteq as eteq
import layr.layr as layr

prog_description = 'Demo lstm model'

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
    parser.add_argument('--load', dest='load', nargs='?', default='models/lstm.onnx',
        help='Filename to load pretrained model (default: models/lstm.onnx)')
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

    model = layr.lstm(x_dim, mem_cell_ct, len(y_list),
        weight_init=layr.unif_xavier_init(1),
        bias_init=layr.unif_xavier_init(1))
    untrained_model = model.deep_clone()
    pretrained_model = model.deep_clone()
    try:
        print('loading ' + args.load)
        pretrained_model = layr.load_layers_file(args.load)[0]
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))

    test_inputs = eteq.variable(np.array(input_val_arr), 'test_input')
    test_outputs = eteq.variable(np.array(y_list), 'test_outputs')

    untrained = tc.slice(untrained_model.connect(test_inputs), 0, 1, 0)
    hiddens = tc.slice(model.connect(test_inputs), 0, 1, 0)
    pretrained = tc.slice(pretrained_model.connect(test_inputs), 0, 1, 0)

    err = tc.reduce_sum(loss(tc.transpose(tc.slice(hiddens, 0, 1, 0)), test_outputs))
    sess.track([untrained, hiddens, pretrained, err])

    train_err = layr.sgd_train(model, test_inputs, test_outputs,
        layr.get_sgd(learning_rate=0.1),
        err_func=lstm_loss)
    sess.track([train_err])

    eteq.optimize(sess, "cfg/optimizations.json")

    start = time.time()
    for cur_iter in range(args.n_train):
        sess.update_target([train_err])

        sess.update_target([hiddens, err])
        print("iter {}: y_pred = {}, loss: {}".format(
            cur_iter, hiddens.get().flatten(), err.get()))

    sess.update_target([untrained, hiddens, pretrained])
    print("expecting = {}".format(np.array(y_list)))
    print("untrained_y_pred = {}".format(untrained.get().flatten()))
    print("trained_y_pred = {}".format(hiddens.get().flatten()))
    print("pretrained_y_pred = {}".format(pretrained.get().flatten()))
    print('training time: {} seconds'.format(time.time() - start))

    try:
        print('saving')
        if layr.save_layers_file(args.save, [model]):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if __name__ == "__main__":
    main(sys.argv[1:])
