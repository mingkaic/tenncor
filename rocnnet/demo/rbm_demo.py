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

class RBMInput:
    def __init__(self, n_visible, n_hidden, n_batch):
        self.x = ead.scalar_variable(0, [n_batch, n_visible])
        self.y = ead.scalar_variable(0, [n_batch, n_hidden])

class RBMStorage:
    def __init__(self, config):
        hidden_layer = rcn.Dense(config.n_hidden, config.n_visible,
            weight_init=rcn.unif_xavier_init(config.xavier_const),
            bias_init=rcn.zero_init(), label="hidden_0")
        weight, hbias = tuple(hidden_layer.get_contents())
        visible_layer = rcn.create_dense(tf.transpose(weight),
            bias=rcn.zero_init()(rcn.Shape([config.n_visible]), "bias"), label="visible_0")
        _, vbias = tuple(visible_layer.get_contents())

        self.hidden_model = rcn.SequentialModel('hidden')
        self.hidden_model.add(hidden_layer)
        self.hidden_model.add(rcn.sigmoid())

        self.visible_model = rcn.SequentialModel('visible')
        self.visible_model.add(visible_layer)
        self.visible_model.add(rcn.sigmoid())

        self.w = weight
        self.visible_bias = vbias
        self.hidden_bias = hbias

        self.delta_w = ead.scalar_variable(0, [config.n_visible, config.n_hidden])
        self.delta_visible_bias = ead.scalar_variable(0, [config.n_visible])
        self.delta_hidden_bias = ead.scalar_variable(0, [config.n_hidden])

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

def bb_init(rbm_in, rbm_storage, config):
    # hidden = sigmoid(X @ W + hB)
    # visible_rec = sigmoid(tf.random.rand_binom_one(hidden) @ W^T + vB)
    # hidden_rec = sigmoid(visible_rec @ W + hB)
    compute_hidden = rbm_storage.hidden_model.connect(rbm_in.x)
    compute_visible = rbm_storage.visible_model.connect(
        tf.random.rand_binom_one(compute_hidden))
    compute_err = config.err_function(rbm_in.x, compute_visible)

    hidden_recon_p = rbm_storage.hidden_model.connect(compute_visible)

    # X^T @ sigmoid(X @ W + hB) - sigmoid(tf.random.rand_binom_one(sigmoid(X @ W + hB)) @ W^T + vB)^T @ sigmoid(sigmoid(tf.random.rand_binom_one(hidden) @ W^T + vB) @ W + hB)
    grad = tf.sub(
        tf.matmul(tf.transpose(rbm_in.x), compute_hidden),
        tf.matmul(tf.transpose(compute_visible), hidden_recon_p))
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
    delta_visible_bias_new = f(rbm_storage.delta_visible_bias, tf.reduce_mean_1d(tf.sub(rbm_in.x, compute_visible), 1))
    delta_hidden_bias_new = f(rbm_storage.delta_hidden_bias, tf.reduce_mean_1d(tf.sub(compute_hidden, hidden_recon_p), 1))

    update_delta_w = (rbm_storage.delta_w, delta_w_new)
    update_delta_visible_bias = (rbm_storage.delta_visible_bias, delta_visible_bias_new)
    update_delta_hidden_bias = (rbm_storage.delta_hidden_bias, delta_hidden_bias_new)
    update_w = (rbm_storage.w, tf.add(rbm_storage.w, delta_w_new))
    update_visible_bias = (rbm_storage.visible_bias,
        tf.add(rbm_storage.visible_bias, delta_visible_bias_new))
    update_hidden_bias = (rbm_storage.hidden_bias,
        tf.add(rbm_storage.hidden_bias, delta_hidden_bias_new))

    return RBMOutput(
        assigns=[
            update_delta_w, update_delta_visible_bias, update_delta_hidden_bias,
            update_w, update_visible_bias, update_hidden_bias,
        ],
        compute_hidden=compute_hidden,
        compute_visible=compute_visible,
        compute_visible_from_hidden=rbm_storage.visible_model.connect(rbm_in.y),
        compute_err=compute_err)

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

        assigns = self.output.assigns
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
        assigns = self.output.assigns

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

IMAGE = 1

config = RBMConfig(
    n_visible=784,
    n_hidden=64,
    learning_rate=0.01,
    momentum=0.95,
    tqdm=tqdm)

rbm = RBM(config, bb_init)
sess = ead.Session()

padewan = RBMTrainee(rbm=rbm, n_batch=10, sess=sess)

master_inputs = RBMInput(config.n_visible, config.n_hidden, n_batch=1)
master = rbm(master_inputs)
image = mnist_images[IMAGE]
sess.track([master.compute_visible])

sess.optimize("cfg/optimizations.rules")

errs = padewan.fit(mnist_images, n_epoches=30)
plt.plot(errs)
plt.show()

def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

master_inputs.x.assign(image.reshape(1,-1))
sess.update_target([master.compute_visible], [master_inputs.x])
image_rec = master.compute_visible.get()

show_digit(image)
show_digit(image_rec)
