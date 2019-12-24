import sys
import time
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow_datasets as tfds

import eteq.tenncor as tc
import eteq.eteq as eteq
import layr.layr as layr

prog_description = 'Demo rbm_trainer'

def mse_errfunc(x, visible_sample_):
    return tc.reduce_mean(tc.square(x - visible_sample_))

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
    parser.add_argument('--load', dest='load', nargs='?', default='models/rbm.onnx',
        help='Filename to load pretrained model (default: models/rbm.onnx)')
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        eteq.seed(args.seedval)
        np.random.seed(args.seedval)

    if args.use_tqdm:
        from tqdm import tqdm
        tq = tqdm
    else:
        tq = None

    n_visible = 784
    n_hidden = 64
    learning_rate = 0.01
    momentum = 0.95

    model = layr.rbm(n_hidden, n_visible,
        weight_init=layr.unif_xavier_init(args.xavier_const),
        bias_init=layr.zero_init())

    untrained = model.deep_clone()
    trained = model.deep_clone()
    try:
        print('loading ' + args.load)
        trained = layr.RBMLayer(*layr.load_layers_file(args.load))
        print('successfully loaded from ' + args.load)
    except Exception as e:
        print(e)
        print('failed to load from "{}"'.format(args.load))

    sess = eteq.Session()
    n_batch = 10

    # ds = tfds.load('mnist', split=tfds.Split.TRAIN, batch_size=n_batch)

    train_input = eteq.Variable([n_batch, n_visible])
    train = layr.rbm_train(model, sess, tc.ETensor(train_input),
        learning_rate=learning_rate,
        discount_factor=momentum,
        err_func=mse_errfunc)

    x = eteq.scalar_variable(0, [1, n_visible])
    xin = tc.ETensor(x)
    genx = model.backward_connect(
        tc.random.rand_binom_one(model.connect(xin)))

    untrained_genx = untrained.backward_connect(
        tc.random.rand_binom_one(untrained.connect(xin)))

    trained_genx = trained.backward_connect(
        tc.random.rand_binom_one(trained.connect(xin)))

    mnist_images = input_data.read_data_sets('MNIST_data/', one_hot=True).train.images

    image = random.choice(mnist_images)
    sess.track([genx, trained_genx, untrained_genx])
    eteq.optimize(sess, eteq.parse_optrules("cfg/optimizations.rules"))

    n_epoches = 30
    shuffle = True
    verbose = True
    n_data = mnist_images.shape[0]

    if n_batch > 0:
        n_batches = n_data // n_batch + (0 if n_data % n_batch == 0 else 1)
    else:
        n_batches = 1

    if shuffle:
        inds = np.arange(n_data)

    errs = []
    for e in range(n_epoches):
        epoch_errs = []

        if shuffle:
            np.random.shuffle(inds)
            mnist_images = mnist_images[inds]

        # ds = tfds.load('mnist', split=tfds.Split.TRAIN, shuffle_files=True, batch_size=10)

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
            batch_x = mnist_images[b * n_batch:(b + 1) * n_batch]
            train_input.assign(batch_x)
            err = train().as_numpy()
            epoch_errs.append(err)

        epoch_errs = np.array(epoch_errs)
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
    sess.update_target([genx, trained_genx, untrained_genx])
    image_rec = genx.get()
    image_rec_trained = trained_genx.get()
    image_rec_untrained = untrained_genx.get()

    show_digit(image, plt)
    show_digit(image_rec, plt)
    show_digit(image_rec_trained, plt)
    show_digit(image_rec_untrained, plt)

    try:
        print('saving')
        if layr.save_layers_file(args.save, [model.fwd(), model.bwd()]):
            print('successfully saved to {}'.format(args.save))
    except Exception as e:
        print(e)
        print('failed to write to "{}"'.format(args.save))

if '__main__' == __name__:
    main(sys.argv[1:])
