from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import ead.tenncor as tenncor
import ead.ead as ead
import rocnnet.rocnnet as rcn


def mse_errfunc(x, visible_sample_):
    return tenncor.reduce_mean(tenncor.square(tenncor.sub(x, visible_sample_)))

class RBMConfig:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function=mse_errfunc,
                 tqdm=None):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.xavier_const=xavier_const
        self.err_function=err_function
        self.tqdm=tqdm

# ==================== demo =======================
from tqdm import tqdm
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

IMAGE = 1

config = RBMConfig(
    n_visible=784,
    n_hidden=64,
    learning_rate=0.01,
    momentum=0.95,
    tqdm=tqdm)

model = rcn.RBM(config.n_hidden, config.n_visible,
    weight_init=rcn.unif_xavier_init(config.xavier_const),
    bias_init=rcn.zero_init(),
    label="demo")
sess = ead.Session()

batch_size = 10

padewan = rcn.BernoulliRBMTrainer(
    model=model,
    sess=sess,
    batch_size=batch_size,
    learning_rate=config.learning_rate,
    discount_factor=config.momentum,
    err_func=config.err_function)

x = ead.scalar_variable(0, [1, config.n_visible])
genx = model.backward_connect(
    tenncor.random.rand_binom_one(model.connect(x)))

image = mnist_images[IMAGE]
sess.track([genx])

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
        if config.tqdm is not None:
            r_batches = config.tqdm(r_batches,
                desc='Epoch: {:d}'.format(e),
                ascii=True,
                file=sys.stdout)
        else:
            print('Epoch: {:d}'.format(e))

    for b in r_batches:
        batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]

        epoch_errs[epoch_errs_ptr] = padewan.train(batch_x)
        epoch_errs_ptr += 1

    if verbose:
        err_mean = epoch_errs.mean()
        if config.tqdm is not None:
            config.tqdm.write('Train error: {:.4f}'.format(err_mean))
            config.tqdm.write('')
        else:
            print('Train error: {:.4f}'.format(err_mean))
            print('')
        sys.stdout.flush()

    errs = np.hstack([errs, epoch_errs])

plt.plot(errs)
plt.show()

def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

x.assign(image.reshape(1,-1))
sess.update_target([genx], [x])
image_rec = genx.get()

show_digit(image)
show_digit(image_rec)
