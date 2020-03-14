import sys
import time
import argparse

import numpy as np

import tenncor as tc

prog_description = 'Demo sgd trainer'

def batch_generate(n, batchsize):
    inbatch = np.random.rand(batchsize * n)
    outbatch = (inbatch[0::2] + inbatch[1::2]) / 2
    return inbatch, outbatch

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
        type=str2bool, nargs='?', const=False, default=True,
        help='Whether to seed or not (default: True)')
    parser.add_argument('--seedval', dest='seedval', type=int, nargs='?', default=int(default_ts),
        help='Random seed value (default: <current time>)')
    parser.add_argument('--nbatch', dest='nbatch', type=int, nargs='?', default=3,
        help='Batch size when training (default: 3)')
    parser.add_argument('--n_train', dest='n_train', type=int, nargs='?', default=3000,
        help='Number of times to train (default: 3000)')
    parser.add_argument('--n_test', dest='n_test', type=int, nargs='?', default=500,
        help='Number of times to test (default: 500)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='models/gd.onnx',
        help='Filename to load pretrained model (default: models/gd.onnx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        tc.seed(args.seedval)
        np.random.seed(args.seedval)

    nunits = 9
    ninput = 10
    noutput = int(ninput / 2)
    nbatch = args.nbatch

    train_input = tc.EVariable([nbatch, ninput])
    train_output = tc.EVariable([nbatch, noutput])
    model = tc.link([
        tc.layer.dense([ninput], [nunits],
            weight_init=tc.unif_xavier_init(),
            bias_init=tc.zero_init()),
        tc.bind(tc.sigmoid),
        tc.layer.dense([nunits], [noutput],
            weight_init=tc.unif_xavier_init(),
            bias_init=tc.zero_init()),
        tc.bind(tc.sigmoid),
    ], train_input)
    train_err = tc.sgd_train(model, train_input, train_output,
        lambda assocs: tc.approx.sgd(assocs, 0.9))
    untrained = model.deep_clone()
    trained = model.deep_clone()
    try:
        print('loading ' + args.load)
        trained = tc.load_layers_file(args.load)[0]
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))

    sess = tc.Session()
    sess.track([train_err])

    testin = tc.EVariable([ninput], label='testin')
    untrained_out = untrained.connect(testin)
    trained_out = model.connect(testin)
    pretrained_out = trained.connect(testin)
    sess.track([untrained_out, trained_out, pretrained_out])

    tc.optimize(sess, "cfg/optimizations.json")

    show_every_n = 500
    start = time.time()
    for i in range(args.n_train):
        batch, batch_out = batch_generate(ninput, nbatch)
        train_input.assign(batch.reshape(nbatch, ninput))
        train_output.assign(batch_out.reshape(nbatch, noutput))
        sess.update_target([train_err])
        if i % show_every_n == show_every_n - 1:
            err = train_err.get()
            print('training {}\ntraining error:\n{}'
                .format(i + 1, err))

    print('training time: {} seconds'.format(time.time() - start))

    untrained_err = 0
    trained_err = 0
    pretrained_err = 0

    for i in range(args.n_test):
        if i % show_every_n == show_every_n - 1:
            print('testing {}'.format(i + 1))

        test_batch, test_batch_out = batch_generate(ninput, 1)
        testin.assign(test_batch)
        sess.update()

        untrained_data = untrained_out.get()
        trained_data = trained_out.get()
        pretrained_data = pretrained_out.get()

        untrained_err += np.mean(abs(untrained_data - test_batch_out))
        trained_err += np.mean(abs(trained_data - test_batch_out))
        pretrained_err += np.mean(abs(pretrained_data - test_batch_out))

    untrained_err /= args.n_test
    trained_err /= args.n_test
    pretrained_err /= args.n_test
    print('untrained mlp error rate: {}%'.format(untrained_err * 100))
    print('trained mlp error rate: {}%'.format(trained_err * 100))
    print('pretrained mlp error rate: {}%'.format(pretrained_err * 100))

    try:
        print('saving')
        if tc.save_layers_file(args.save, [model]):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if '__main__' == __name__:
    main(sys.argv[1:])
