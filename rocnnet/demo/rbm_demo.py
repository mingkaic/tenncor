from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import ead.tenncor as tf
import ead.ead as ead
import rocnnet.rocnnet as rcn


def mse_errfunc(x, compute_visible):
    return tf.reduce_mean(tf.square(tf.sub(x, compute_visible)))

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
def bbernoulli_optimizer(varerrs):
    out = []
    for (var, err) in varerrs:
        momentum = ead.scalar_variable(0.0, err.shape(), label=str(var) + "_momentum")
        next_momentum = tf.add(
            tf.mul(
                ead.scalar_constant(config.momentum, momentum.shape()),
                momentum
            ),
            tf.mul(
                ead.scalar_constant(config.learning_rate *
                    (1 - config.momentum) / err.shape()[0], err.shape()),
                err
            ))
        next_var = tf.add(var, next_momentum)
        out.append((momentum, next_momentum))
        out.append((var, next_var))
    return out

class RBMOutput:
    def __init__(self, assigns, compute_hidden, compute_visible, compute_visible_from_hidden, compute_err):
        self.assigns = assigns
        self.compute_hidden = compute_hidden
        self.compute_visible = compute_visible
        self.compute_visible_from_hidden = compute_visible_from_hidden
        self.compute_err = compute_err
        assert self.assigns is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None
        assert self.compute_err is not None

def bb_init(x, y, rbm, err_function):
    # hidden = sigmoid(X @ W + hB)
    # visible_rec = sigmoid(tf.random.rand_binom_one(hidden) @ W^T + vB)
    # hidden_rec = sigmoid(visible_rec @ W + hB)
    compute_hidden = rbm.connect(x)
    compute_visible = rbm.backward_connect(
        tf.random.rand_binom_one(compute_hidden))
    compute_err = err_function(x, compute_visible)

    hidden_recon_p = rbm.connect(compute_visible)

    # X^T @ sigmoid(X @ W + hB) - sigmoid(tf.random.rand_binom_one(sigmoid(X @ W + hB)) @ W^T + vB)^T @ sigmoid(sigmoid(tf.random.rand_binom_one(hidden) @ W^T + vB) @ W + hB)
    grad_w = tf.sub(
        tf.matmul(tf.transpose(x), compute_hidden),
        tf.matmul(tf.transpose(compute_visible), hidden_recon_p))
    grad_vb = tf.reduce_mean_1d(tf.sub(x, compute_visible), 1)
    grad_hb = tf.reduce_mean_1d(tf.sub(compute_hidden, hidden_recon_p), 1)
    # grad_w is derivative of ? with respect to rbm.w

    (w, hbias, vbias) = tuple(rbm.get_contents())

    return RBMOutput(
        assigns=bbernoulli_optimizer([
            (w, grad_w),
            (hbias, grad_hb),
            (vbias, grad_vb),
        ]),
        compute_hidden=compute_hidden,
        compute_visible=compute_visible,
        compute_visible_from_hidden=rbm.backward_connect(y),
        compute_err=compute_err)

class RBMTrainee:
    def __init__(self, config, init, rbm, n_batch, sess):
        self.config = config
        self.rbm = rbm
        self.n_batch = n_batch
        self.x = ead.scalar_variable(0, [n_batch, config.n_visible])
        self.y = ead.scalar_variable(0, [n_batch, config.n_hidden])
        output = init(self.x, self.y, rbm, config.err_function)
        self.sess = sess

        self.assigns = output.assigns
        self.compute_err = output.compute_err
        sess.track([
            output.compute_visible,
            output.compute_hidden,
            output.compute_err,
        ] + [src for _, src in self.assigns])

    def fit(self,
            data_x,
            n_epoches=10,
            shuffle=True,
            verbose=True):
        assert n_epoches > 0

        n_data = data_x.shape[0]

        if self.n_batch > 0:
            n_batches = n_data // self.n_batch + (0 if n_data % self.n_batch == 0 else 1)
        else:
            n_batches = 1

        if shuffle:
            data_x_cpy = data_x.copy()
            inds = np.arange(n_data)
        else:
            data_x_cpy = data_x

        errs = []

        for e in range(n_epoches):
            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches)

            if verbose:
                if self.config.tqdm is not None:
                    r_batches = self.config.tqdm(r_batches,
                        desc='Epoch: {:d}'.format(e),
                        ascii=True,
                        file=sys.stdout)
                else:
                    print('Epoch: {:d}'.format(e))

            for b in r_batches:
                batch_x = data_x_cpy[b * self.n_batch:(b + 1) * self.n_batch]

                self.x.assign(batch_x)

                self.sess.update_target([src for _, src in self.assigns], [self.x])
                for dest, src in self.assigns:
                    dest.assign(src.get())
                self.sess.update_target([self.compute_err], [dest for dest, _ in self.assigns])
                batch_err = self.compute_err.get()

                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self.config.tqdm is not None:
                    self.config.tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self.config.tqdm.write('')
                else:
                    print('Train error: {:.4f}'.format(err_mean))
                    print('')
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

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

padewan = RBMTrainee(config, init=bb_init, rbm=rbm, n_batch=10, sess=sess)

master_x = ead.scalar_variable(0, [1, config.n_visible])
master_hidden = rbm.connect(master_x)
master_visible = rbm.backward_connect(
    tf.random.rand_binom_one(master_hidden))

image = mnist_images[IMAGE]
sess.track([master_visible])

sess.optimize("cfg/optimizations.rules")

errs = padewan.fit(mnist_images, n_epoches=30)
plt.plot(errs)
plt.show()

def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

master_x.assign(image.reshape(1,-1))
sess.update_target([master_visible], [master_x])
image_rec = master_visible.get()

show_digit(image)
show_digit(image_rec)
