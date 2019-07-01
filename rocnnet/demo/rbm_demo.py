import sys
import time
import argparse

import numpy as np

import ead.tenncor as tc
import ead.ead as ead
import rocnnet.rocnnet as rcn
import dbg.grpc_dbg as dbg

from rocnnet.demo.data.load_mnist import load_mnist
from rocnnet.demo.data.image_out import mnist_imageout

prog_description = 'Demo rbm_trainer'
default_n_sample = 10

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
    parser.add_argument('--train', dest='train',
        type=str2bool, nargs='?', const=False, default=True,
        help='Whether to train or not (default: True)')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, nargs='?', default=0.1,
        help='Learning rate (default: 0.1)')
    parser.add_argument('--seedval', dest='seedval', type=int, nargs='?', default=int(default_ts),
        help='Random seed value (default: <current time>)')
    parser.add_argument('--n_cont_div', dest='n_cont_div', type=int, nargs='?', default=15,
        help='Number of contrastive divergence steps (default: 15)')
    parser.add_argument('--n_batch', dest='n_batch', type=int, nargs='?', default=50,
        help='Batch size when training (default: 50)')
    parser.add_argument('--n_hidden', dest='n_hidden', type=int, nargs='?', default=50,
        help='Size of hidden layer (default: 50)')
    parser.add_argument('--n_test_chain', dest='n_test_chain', type=int, nargs='?', default=20,
        help='Size of test chain (default: 20)')
    parser.add_argument('--n_train', dest='n_train', type=int, nargs='?', default=10,
        help='Number of times to train (default: 10)')
    parser.add_argument('--n_sample', dest='n_sample', type=int, nargs='?', default=default_n_sample,
        help='Number of times to sample (default: {})'.format(default_n_sample))
    parser.add_argument('--cache', dest='cache_file', nargs='?', default='/tmp/rbmcache.pbx',
        help='Filename to cache model (default: /tmp/rbmcache.pbx')
    parser.add_argument('--imgdir', dest='outdir', nargs='?', default='',
        help='Directory to save output images (default: <blank>)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        ead.seed(args.seedval)
        np.random.seed(args.seedval)

    (training_x, training_y), (valid_x, valid_y), (testing_x, testing_y) = load_mnist()

    n_in = training_x.shape[1]
    n_hidden = args.n_hidden

    brain = rcn.get_rbm(n_in, [n_hidden], 'brain')
    try:
        with open(args.cache_file, 'rb') as f:
            print('loading')
            brain = brain.parse_from_string(f.read())
    except:
        pass

    # test
    plot_every = 1000
    n_test_input = testing_x.shape[1]
    n_test_sample = testing_x.shape[0]
    def idx_generate():
        return np.random.randint(0, n_test_sample - args.n_test_chain)

    # train
    # sess = ead.Session()
    sess = dbg.get_isess(request_dur=5000, stream_dur=100000)

    testin = ead.variable(np.zeros([args.n_test_chain, n_test_input], dtype=float), "testin")
    test_generated_in = brain.reconstruct_visible(testin, [tc.sigmoid])
    sess.track([test_generated_in])

    if args.train:
        n_data = training_x.shape[0]
        n_training_batches = n_data / args.n_batch

        persistent = ead.variable(
            np.zeros([args.n_batch, args.n_hidden], dtype=float), 'persistent')
        trainer = rcn.RBMTrainer(brain, [tc.sigmoid], sess, persistent, args.n_batch,
            args.learning_rate, args.n_cont_div)
        sess.optimize("cfg/optimizations.rules")

        for i in range(args.n_train):
            mean_cost = 0
            for j in range(n_training_batches):
                batch = training_x[j:j+args.n_batch]
                mean_cost = mean_cost + trainer.train(list(batch.flatten()))
                print('completed batch {}, mean cost {}'.format(j, mean_cost))

            print("training epoch {}, cost is {}".format(i, mean_cost))
            # save in case of problems
            print('saving to cache')
            brain.serialize_to_file(trainer.cost(), args.cache_file)

    output_chains = []
    for i in range(args.n_sample):
        print("... plotting sample {}".format(i))
        idx = idx_generate()
        test_sample = testing_x[idx:idx + args.n_test_chain]
        for j in range(plot_every):
            testin.assign(test_sample)
            sess.update([testin])
            test_sample = test_generated_in.get()

        output_chains.append(test_sample)

    mnist_imageout(args.outdir, output_chains, testin.shape(), args.n_test_chain, args.n_sample)

if '__main__' == __name__:
    main(sys.argv[1:])
