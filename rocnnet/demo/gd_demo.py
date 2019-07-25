import sys
import time
import argparse

import numpy as np

import ead.tenncor as tc
import ead.ead as ead
import rocnnet.rocnnet as rcn

prog_description = 'Demo mlp_trainer using sgd'

def batch_generate(n, batchsize):
    total = n * batchsize
    return np.random.rand(total)

def avgevry2(indata):
    return (indata[0::2] + indata[1::2]) / 2

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
    parser.add_argument('--n_batch', dest='n_batch', type=int, nargs='?', default=3,
        help='Batch size when training (default: 3)')
    parser.add_argument('--n_train', dest='n_train', type=int, nargs='?', default=3000,
        help='Number of times to train (default: 3000)')
    parser.add_argument('--n_test', dest='n_test', type=int, nargs='?', default=500,
        help='Number of times to test (default: 500)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='rocnnet/pretrained/gdmodel.pbx',
        help='Filename to load pretrained model (default: rocnnet/pretrained/gdmodel.pbx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        ead.seed(args.seedval)
        np.random.seed(args.seedval)

    n_in = 10
    n_out = n_in / 2
    n_outs = [9, n_out]

    nonlins = [tc.sigmoid, tc.sigmoid]

    brain = rcn.get_mlp(n_in, n_outs, 'brain')
    untrained_brain = brain.copy()
    try:
        with open(args.load, 'rb') as f:
            print('loading')
            pretrained_brain = brain.parse_from_string(f.read())
            print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))
        pretrained_brain = brain.copy()

    sess = ead.Session()
    n_batch = args.n_batch
    show_every_n = 500
    trainer = rcn.MLPTrainer(brain, nonlins, sess,
        rcn.get_sgd(0.9), n_batch)

    testin = ead.variable(np.zeros([n_in], dtype=float), 'testin')
    untrained_out = untrained_brain.forward(testin, nonlins)
    trained_out = brain.forward(testin, nonlins)
    pretrained_out = pretrained_brain.forward(testin, nonlins)
    sess.track([
        untrained_out,
        trained_out,
        pretrained_out,
    ])
    sess.optimize("cfg/optimizations.rules")

    start = time.time()
    for i in range(args.n_train):
        if i % show_every_n == show_every_n - 1:
            trained_derr = trainer.error().get()
            print('training {}\ntraining error:\n{}'
                .format(i + 1, trained_derr))
        batch = batch_generate(n_in, n_batch)
        batch_out = avgevry2(batch)
        trainer.train(batch, batch_out)

    print('training time: {} seconds'.format(time.time() - start))

    untrained_err = 0
    trained_err = 0
    pretrained_err = 0

    for i in range(args.n_test):
        if i % show_every_n == show_every_n - 1:
            print('testing {}'.format(i + 1))

        test_batch = batch_generate(n_in, 1)
        test_batch_out = avgevry2(test_batch)
        testin.assign(test_batch)
        sess.update([testin])

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
        if trainer.serialize_to_file(args.save):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if '__main__' == __name__:
    main(sys.argv[1:])
