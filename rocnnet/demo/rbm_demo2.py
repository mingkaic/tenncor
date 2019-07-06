from __future__ import print_function

import sys

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import ead.tenncor as tf
import ead.ead as ead
import rocnnet.rocnnet as rcn


def sample_bernoulli(probs):
    return tf.random.rand_binom_one(probs)

# def sample_gaussian(x, sigma):
#     return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)


# def cosine_errfunc(x, compute_visible):
#     x1_norm = tf.nn.l2_normalize(x, 1)
#     x2_norm = tf.nn.l2_normalize(compute_visible, 1)
#     cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
#     return tf.acos(cos_val) / tf.constant(np.pi)

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

class RBMInput:
    def __init__(self, n_visible, n_hidden, n_batch):
        self.x = ead.scalar_variable(0, [n_batch, n_visible])
        self.y = ead.scalar_variable(0, [n_batch, n_hidden])

class RBMStorage:
    def __init__(self, config):
        self.w = rcn.variable_from_init(
            rcn.unif_xavier_init(config.xavier_const),
            [config.n_visible, config.n_hidden])
        self.visible_bias = ead.scalar_variable(0, [config.n_visible])
        self.hidden_bias = ead.scalar_variable(0, [config.n_hidden])

        self.delta_w = ead.scalar_variable(0, [config.n_visible, config.n_hidden])
        self.delta_visible_bias = ead.scalar_variable(0, [config.n_visible])
        self.delta_hidden_bias = ead.scalar_variable(0, [config.n_hidden])

class RBMOutput:
    def __init__(self, update_weights, update_deltas, compute_hidden, compute_visible, compute_visible_from_hidden, compute_err):
        self.update_weights = update_weights
        self.update_deltas = update_deltas
        self.compute_hidden = compute_hidden
        self.compute_visible = compute_visible
        self.compute_visible_from_hidden = compute_visible_from_hidden
        self.compute_err = compute_err
        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None
        assert self.compute_err is not None

def bb_init(rbm_in, rbm_storage, config):
    # hidden = sigmoid(X @ W + hB)
    # visible_rec = sigmoid(sample_bernoulli(hidden) @ W^T + vB)
    # hidden_rec = sigmoid(visible_rec @ W + hB)
    hidden_p = tf.sigmoid(tf.nn.fully_connect(
        [rbm_in.x], [rbm_storage.w],
        rbm_storage.hidden_bias))
    visible_recon_p = tf.sigmoid(tf.nn.fully_connect(
        [sample_bernoulli(hidden_p)], [tf.transpose(rbm_storage.w)],
        rbm_storage.visible_bias))
    hidden_recon_p = tf.sigmoid(tf.nn.fully_connect(
        [visible_recon_p], [rbm_storage.w],
        rbm_storage.hidden_bias))

    # X^T @ sigmoid(X @ W + hB) - sigmoid(sample_bernoulli(sigmoid(X @ W + hB)) @ W^T + vB)^T @ sigmoid(sigmoid(sample_bernoulli(hidden) @ W^T + vB) @ W + hB)
    grad = tf.sub(
        tf.matmul(
            tf.transpose(rbm_in.x),
            hidden_p),
        tf.matmul(
            tf.transpose(visible_recon_p),
            hidden_recon_p))
    # grad is derivative of ? with respect to rbm_in.w

    def f(x_old, x_new):
        return tf.add(
            tf.mul(
                ead.scalar_constant(config.momentum, x_old.shape()),
                x_old),
            tf.prod([
                ead.scalar_constant(config.learning_rate, x_new.shape()),
                x_new,
                ead.scalar_constant((1 - config.momentum) / x_new.shape()[0], x_new.shape())
            ]))

    # momentum
    delta_w_new = f(rbm_storage.delta_w, grad)
    delta_visible_bias_new = f(rbm_storage.delta_visible_bias, tf.reduce_mean_1d(tf.sub(rbm_in.x, visible_recon_p), 1))
    delta_hidden_bias_new = f(rbm_storage.delta_hidden_bias, tf.reduce_mean_1d(tf.sub(hidden_p, hidden_recon_p), 1))

    update_delta_w = (rbm_storage.delta_w, delta_w_new)
    update_delta_visible_bias = (rbm_storage.delta_visible_bias, delta_visible_bias_new)
    update_delta_hidden_bias = (rbm_storage.delta_hidden_bias, delta_hidden_bias_new)

    update_w = (rbm_storage.w, tf.add(rbm_storage.w, delta_w_new))
    update_visible_bias = (rbm_storage.visible_bias,
        tf.add(rbm_storage.visible_bias, delta_visible_bias_new))
    update_hidden_bias = (rbm_storage.hidden_bias,
        tf.add(rbm_storage.hidden_bias, delta_hidden_bias_new))

    compute_hidden = tf.sigmoid(tf.nn.fully_connect(
        [rbm_in.x], [rbm_storage.w],
        rbm_storage.hidden_bias))
    compute_visible = tf.sigmoid(tf.nn.fully_connect(
        [compute_hidden], [tf.transpose(rbm_storage.w)],
        rbm_storage.visible_bias))
    return RBMOutput(
        update_deltas=[update_delta_w, update_delta_visible_bias, update_delta_hidden_bias],
        update_weights=[update_w, update_visible_bias, update_hidden_bias],
        compute_hidden=compute_hidden,
        compute_visible=compute_visible,
        compute_visible_from_hidden=tf.sigmoid(tf.nn.fully_connect(
            [rbm_in.y], [tf.transpose(rbm_storage.w)],
            rbm_storage.visible_bias)),
        compute_err=config.err_function(rbm_in.x, compute_visible))

# def gr_init(rbm_in, rbm_storage, config, sample_visible=False, sigma=1):
#     hidden_p = tf.nn.sigmoid(tf.matmul(rbm_in.x, rbm_storage.w) + rbm_storage.hidden_bias)
#     visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(rbm_storage.w)) + rbm_storage.visible_bias

#     if sample_visible:
#         visible_recon_p = sample_gaussian(visible_recon_p, sigma)

#     hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, rbm_storage.w) + rbm_storage.hidden_bias)

#     positive_grad = tf.matmul(tf.transpose(rbm_in.x), hidden_p)
#     negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

#     def f(x_old, x_new):
#         return config.momentum * x_old +\
#             config.learning_rate * x_new * (1 - config.momentum) / tf.to_float(tf.shape(x_new)[0])

#     delta_w_new = f(rbm_storage.delta_w, positive_grad - negative_grad)
#     delta_visible_bias_new = f(rbm_storage.delta_visible_bias, tf.reduce_mean(rbm_in.x - visible_recon_p, 0))
#     delta_hidden_bias_new = f(rbm_storage.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

#     update_delta_w = rbm_storage.delta_w.assign(delta_w_new)
#     update_delta_visible_bias = rbm_storage.delta_visible_bias.assign(delta_visible_bias_new)
#     update_delta_hidden_bias = rbm_storage.delta_hidden_bias.assign(delta_hidden_bias_new)

#     update_w = rbm_storage.w.assign(rbm_storage.w + delta_w_new)
#     update_visible_bias = rbm_storage.visible_bias.assign(rbm_storage.visible_bias + delta_visible_bias_new)
#     update_hidden_bias = rbm_storage.hidden_bias.assign(rbm_storage.hidden_bias + delta_hidden_bias_new)

#     compute_hidden = tf.nn.sigmoid(tf.matmul(rbm_in.x, rbm_storage.w) + rbm_storage.hidden_bias)
#     compute_visible = tf.matmul(compute_hidden, tf.transpose(rbm_storage.w)) + rbm_storage.visible_bias
#     return RBMOutput(
#         update_deltas=[update_delta_w, update_delta_visible_bias, update_delta_hidden_bias],
#         update_weights=[update_w, update_visible_bias, update_hidden_bias],
#         compute_hidden=compute_hidden,
#         compute_visible=compute_visible,
#         compute_visible_from_hidden=tf.matmul(rbm_in.y, tf.transpose(rbm_storage.w)) + rbm_storage.visible_bias,
#         compute_err=config.err_function(rbm_in.x, compute_visible))

class RBM:
    def __init__(self, config, initialize):
        self.config = config
        self.initialize = initialize
        self.storage = RBMStorage(config)

    def __call__(self, inputs):
        return self.initialize(inputs, self.storage, self.config)

class RBMTrainee:
    def __init__(self, rbm, n_batch, sess):
        self.rbm = rbm
        self.n_batch = n_batch
        self.input = RBMInput(config.n_visible, config.n_hidden, n_batch=n_batch)
        self.output = rbm(self.input)
        self.sess = sess

        assigns = self.output.update_weights + self.output.update_deltas
        sess.track([
            self.output.compute_visible,
            self.output.compute_hidden,
            self.output.compute_err] + [src for _, src in assigns])

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
        assigns = self.output.update_weights + self.output.update_deltas

        for e in range(n_epoches):
            epoch_errs = np.zeros((n_batches,))
            epoch_errs_ptr = 0

            if shuffle:
                np.random.shuffle(inds)
                data_x_cpy = data_x_cpy[inds]

            r_batches = range(n_batches)

            if verbose:
                if self.rbm.config.tqdm is not None:
                    r_batches = self.rbm.config.tqdm(r_batches,
                        desc='Epoch: {:d}'.format(e),
                        ascii=True,
                        file=sys.stdout)
                else:
                    print('Epoch: {:d}'.format(e))

            for b in r_batches:
                batch_x = data_x_cpy[b * self.n_batch:(b + 1) * self.n_batch]

                self.input.x.assign(batch_x)

                self.sess.update_target([src for _, src in assigns], [self.input.x])
                for dest, src in assigns:
                    dest.assign(src.get())
                self.sess.update_target([self.output.compute_err], [dest for dest, _ in assigns])
                batch_err = self.output.compute_err.get()

                epoch_errs[epoch_errs_ptr] = batch_err
                epoch_errs_ptr += 1

            if verbose:
                err_mean = epoch_errs.mean()
                if self.rbm.config.tqdm is not None:
                    self.rbm.config.tqdm.write('Train error: {:.4f}'.format(err_mean))
                    self.rbm.config.tqdm.write('')
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

config = RBMConfig(
    n_visible=784,
    n_hidden=64,
    learning_rate=0.01,
    momentum=0.95,
    tqdm=tqdm)

rbm = RBM(config, bb_init)
sess = ead.Session()

padewan = RBMTrainee(rbm=rbm, n_batch=10, sess=sess)
errs = padewan.fit(mnist_images, n_epoches=30)
plt.plot(errs)
plt.show()

IMAGE = 1

def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

master_inputs = RBMInput(config.n_visible, config.n_hidden, n_batch=1)
master = rbm(master_inputs)
image = mnist_images[IMAGE]
sess.track([master.compute_visible])
master_inputs.x.assign(image.reshape(1,-1))
sess.update_target([master.compute_visible], [master_inputs.x])
image_rec = master.compute_visible.get()

show_digit(image)
show_digit(image_rec)
