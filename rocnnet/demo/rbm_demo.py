import sys
import time
import random
import argparse

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

import ead.tenncor as tc
import ead.ead as ead
import rocnnet.rocnnet as rcn

prog_description = 'Demo rbm_trainer'

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def mse_errfunc(x, visible_sample_):
    return tc.reduce_mean(tc.square(tc.sub(x, visible_sample_)))

def show_digit(x, plt):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

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
    parser.add_argument('--init_const', dest='xavier_const', type=float, nargs='?', default=1.0,
        help='Xavier constant for initializing weight (default: 1.0)')
    parser.add_argument('--use_tqdm', dest='use_tqdm',
        type=str2bool, nargs='?', const=False, default=True,
        help='Whether to use tqdm (default: 100)')
    parser.add_argument('--save', dest='save', nargs='?', default='',
        help='Filename to save model (default: <blank>)')
    parser.add_argument('--load', dest='load', nargs='?', default='models/rbmmodel.pbx',
        help='Filename to load pretrained model (default: models/rbmmodel.pbx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        ead.seed(args.seedval)
        np.random.seed(args.seedval)

    if args.use_tqdm:
        from tqdm import tqdm
        tq = tqdm
    else:
        tq = None

    mnist_images = mnist.train.images

    n_visible = 784
    n_hidden = 64
    learning_rate = 0.01
    momentum = 0.95

    model = rcn.RBM(n_hidden, n_visible,
        weight_init=rcn.unif_xavier_init(args.xavier_const),
        bias_init=rcn.zero_init(),
        label="demo")

    untrained = model.clone()
    try:
        print('loading ' + args.load)
        trained = rcn.load_file_rbmmodel(args.load, "demo")
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))
        trained = model.clone()

    sess = ead.Session()
    batch_size = 10

    trainer = rcn.BernoulliRBMTrainer(
        model=model,
        sess=sess,
        batch_size=batch_size,
        learning_rate=learning_rate,
        discount_factor=momentum,
        err_func=mse_errfunc)

    x = ead.scalar_variable(0, [1, n_visible])
    genx = model.backward_connect(
        tc.random.rand_binom_one(model.connect(x)))

    untrained_genx = untrained.backward_connect(
        tc.random.rand_binom_one(untrained.connect(x)))

    trained_genx = trained.backward_connect(
        tc.random.rand_binom_one(trained.connect(x)))

    image = random.choice(mnist_images)
    sess.track([genx, trained_genx, untrained_genx])

    sess.optimize("cfg/optimizations.rules")

    n_epoches = 30
    shuffle = True
    verbose = True
    n_data = mnist_images.shape[0]

    if batch_size > 0:
        n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
    else:
        n_batches = 1

    if shuffle:
        data_x_cpy = mnist_images.copy()
        inds = np.arange(n_data)
    else:
        data_x_cpy = mnist_images

    errs = []
    for e in range(n_epoches):
        epoch_errs = np.zeros((n_batches,))
        epoch_errs_ptr = 0

        if shuffle:
            np.random.shuffle(inds)
            data_x_cpy = data_x_cpy[inds]

        r_batches = range(n_batches)

        if verbose:
            if tq is not None:
                r_batches = tq(r_batches,
                    desc='Epoch: {:d}'.format(e),
                    ascii=True,
                    file=sys.stdout)
            else:
                print('Epoch: {:d}'.format(e))

        for b in r_batches:
            batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]

            epoch_errs[epoch_errs_ptr] = trainer.train(batch_x)
            epoch_errs_ptr += 1

        if verbose:
            err_mean = epoch_errs.mean()
            if tq is not None:
                tq.write('Train error: {:.4f}'.format(err_mean))
                tq.write('')
            else:
                print('Train error: {:.4f}'.format(err_mean))
                print('')
            sys.stdout.flush()

        errs = np.hstack([errs, epoch_errs])

    plt.plot(errs)
    plt.show()

    x.assign(image.reshape(1,-1))
    sess.update_target([genx, trained_genx, untrained_genx], [x])
    image_rec = genx.get()
    image_rec_trained = trained_genx.get()
    image_rec_untrained = untrained_genx.get()

    show_digit(image, plt)
    show_digit(image_rec, plt)
    show_digit(image_rec_trained, plt)
    show_digit(image_rec_untrained, plt)

    try:
        print('saving')
        if model.save_file(args.save):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if '__main__' == __name__:
    main(sys.argv[1:])
