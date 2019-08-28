from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import ead.tenncor as tf
import ead.ead as ead
import rocnnet.rocnnet as rcn


def mse_errfunc(x, visible_sample_):
    return tf.reduce_mean(tf.square(tf.sub(x, visible_sample_)))

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

# next_momentum = prev_momentum * momentum +
#                 learning_rate * (1 - momentum) * err
# next = prev + next_momentum
def bbernoulli_optimizer(varerrs, learning_rate, discount_factor):
    out = []
    for (var, err) in varerrs:
        momentum = ead.scalar_variable(0.0, err.shape(), label=str(var) + "_momentum")
        next_momentum = tf.add(
            tf.mul(
                ead.scalar_constant(discount_factor, momentum.shape()),
                momentum
            ),
            tf.mul(
                ead.scalar_constant(learning_rate *
                    (1 - discount_factor) / err.shape()[0], err.shape()),
                err
            ))
        next_var = tf.add(var, next_momentum)
        out.append((momentum, next_momentum))
        out.append((var, next_var))
    return out

class RBMOutput:
    def __init__(self, assigns, hidden_sample_, visible_sample_, compute_visible_from_hidden, compute_err):
        self.assigns = assigns
        self.hidden_sample_ = hidden_sample_
        self.visible_sample_ = visible_sample_
        self.compute_visible_from_hidden = compute_visible_from_hidden
        self.compute_err = compute_err
        assert self.assigns is not None
        assert self.hidden_sample_ is not None
        assert self.visible_sample_ is not None
        assert self.compute_visible_from_hidden is not None
        assert self.compute_err is not None

class BernoulliRBMTrainer:
    def __init__(self, rbm, sess, batch_size,
        learning_rate, momentum, err_function):
        self.rbm = rbm
        self.batch_size = batch_size
        self.visible_ = ead.scalar_variable(0, [batch_size, rbm.get_ninput()])
        self.expect_hidden_ = ead.scalar_variable(0, [batch_size, rbm.get_noutput()])
        # hidden = sigmoid(X @ W + hB)
        # visible_rec = sigmoid(tf.random.rand_binom_one(hidden) @ W^T + vB)
        # hidden_rec = sigmoid(visible_rec @ W + hB)
        hidden_sample_ = rbm.connect(self.visible_)
        visible_sample_ = rbm.backward_connect(
            tf.random.rand_binom_one(hidden_sample_))

        hidden_recon_p = rbm.connect(visible_sample_)

        # X^T @ sigmoid(X @ W + hB) - sigmoid(tf.random.rand_binom_one(sigmoid(X @ W + hB)) @ W^T + vB)^T @ sigmoid(sigmoid(tf.random.rand_binom_one(hidden) @ W^T + vB) @ W + hB)
        grad_w = tf.sub(
            tf.matmul(tf.transpose(self.visible_), hidden_sample_),
            tf.matmul(tf.transpose(visible_sample_), hidden_recon_p))
        grad_hb = tf.reduce_mean_1d(tf.sub(hidden_sample_, hidden_recon_p), 1)
        grad_vb = tf.reduce_mean_1d(tf.sub(self.visible_, visible_sample_), 1)
        # grad_w is derivative of ? with respect to rbm.w

        (w, hbias, vbias) = tuple(rbm.get_contents())

        error_ = err_function(self.visible_, visible_sample_)
        output = RBMOutput(
            assigns=bbernoulli_optimizer([
                (w, grad_w),
                (hbias, grad_hb),
                (vbias, grad_vb),
            ], learning_rate, momentum),
            hidden_sample_=hidden_sample_,
            visible_sample_=visible_sample_,
            compute_visible_from_hidden=rbm.backward_connect(self.expect_hidden_),
            compute_err=error_)
        self.sess = sess

        self.assigns = output.assigns
        self.compute_err = output.compute_err
        sess.track([
            output.visible_sample_,
            output.hidden_sample_,
            output.compute_err,
        ] + [src for _, src in self.assigns])

    def train(self, batch_x):
        self.visible_.assign(batch_x)
        self.sess.update_target([src for _, src in self.assigns], [self.visible_])
        for dest, src in self.assigns:
            dest.assign(src.get())
        self.sess.update_target([self.compute_err], [dest for dest, _ in self.assigns])
        return self.compute_err.get()

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

rbm = rcn.RBM(config.n_hidden, config.n_visible,
    weight_init=rcn.unif_xavier_init(config.xavier_const),
    bias_init=rcn.zero_init(),
    label="demo")
sess = ead.Session()

batch_size = 10

# padewan = BernoulliRBMTrainer(rbm=rbm, sess=sess,
#     batch_size=batch_size,
#     learning_rate=config.learning_rate,
#     momentum=config.momentum,
#     err_function=config.err_function)

padewan = rcn.BernoulliRBMTrainer(
    model=rbm,
    sess=sess,
    batch_size=batch_size,
    learning_rate=config.learning_rate,
    discount_factor=config.momentum,
    err_func=config.err_function)

x = ead.scalar_variable(0, [1, config.n_visible])
genx = rbm.backward_connect(
    tf.random.rand_binom_one(rbm.connect(x)))

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
