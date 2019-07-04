from __future__ import print_function

import sys

import tenncor as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def tf_xavier_init(fan_in, fan_out, const=1.0, dtype=np.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)

def sample_bernoulli(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

def sample_gaussian(x, sigma):
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)


def cosine_errfunc(x, compute_visible):
    x1_norm = tf.nn.l2_normalize(x, 1)
    x2_norm = tf.nn.l2_normalize(compute_visible, 1)
    cos_val = tf.reduce_mean(tf.reduce_sum(tf.mul(x1_norm, x2_norm), 1))
    return tf.acos(cos_val) / tf.constant(np.pi)

def mse_errfunc(x, compute_visible):
    return tf.reduce_mean(tf.square(x - compute_visible))


class RBMConfig:
    def __init__(self,
                 n_visible,
                 n_hidden,
                 learning_rate=0.01,
                 momentum=0.95,
                 xavier_const=1.0,
                 err_function=mse_errfunc,
                 tqdm=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.xavier_const=xavier_const
        self.err_function=err_function
        self.tqdm=tqdm

class RBMInput:
    def __init__(self, config):
        self.x = tf.placeholder(tf.float32, [None, config.n_visible])
        self.y = tf.placeholder(tf.float32, [None, config.n_hidden])

        self.w = tf.Variable(tf_xavier_init(config.n_visible, config.n_hidden, const=config.xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([config.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([config.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([config.n_visible, config.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([config.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([config.n_hidden]), dtype=tf.float32)

class RBMOutput:
    def __init__(self, update_weights, update_deltas, compute_hidden, compute_visible, compute_visible_from_hidden):
        self.update_weights = update_weights
        self.update_deltas = update_deltas
        self.compute_hidden = compute_hidden
        self.compute_visible = compute_visible
        self.compute_visible_from_hidden = compute_visible_from_hidden
        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.compute_visible_from_hidden is not None

def bb_init(rbm_in, config):
    hidden_p = tf.nn.sigmoid(tf.matmul(rbm_in.x, rbm_in.w) + rbm_in.hidden_bias)
    visible_recon_p = tf.nn.sigmoid(tf.matmul(sample_bernoulli(hidden_p), tf.transpose(rbm_in.w)) + rbm_in.visible_bias)
    hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, rbm_in.w) + rbm_in.hidden_bias)

    positive_grad = tf.matmul(tf.transpose(rbm_in.x), hidden_p)
    negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

    def f(x_old, x_new):
        return config.momentum * x_old +\
               config.learning_rate * x_new * (1 - config.momentum) / tf.to_float(tf.shape(x_new)[0])

    delta_w_new = f(rbm_in.delta_w, positive_grad - negative_grad)
    delta_visible_bias_new = f(rbm_in.delta_visible_bias, tf.reduce_mean(rbm_in.x - visible_recon_p, 0))
    delta_hidden_bias_new = f(rbm_in.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

    update_delta_w = rbm_in.delta_w.assign(delta_w_new)
    update_delta_visible_bias = rbm_in.delta_visible_bias.assign(delta_visible_bias_new)
    update_delta_hidden_bias = rbm_in.delta_hidden_bias.assign(delta_hidden_bias_new)

    update_w = rbm_in.w.assign(rbm_in.w + delta_w_new)
    update_visible_bias = rbm_in.visible_bias.assign(rbm_in.visible_bias + delta_visible_bias_new)
    update_hidden_bias = rbm_in.hidden_bias.assign(rbm_in.hidden_bias + delta_hidden_bias_new)

    compute_hidden = tf.nn.sigmoid(tf.matmul(rbm_in.x, rbm_in.w) + rbm_in.hidden_bias)
    return RBMOutput(
        update_deltas=[update_delta_w, update_delta_visible_bias, update_delta_hidden_bias],
        update_weights=[update_w, update_visible_bias, update_hidden_bias],
        compute_hidden=compute_hidden,
        compute_visible=tf.nn.sigmoid(tf.matmul(compute_hidden, tf.transpose(rbm_in.w)) + rbm_in.visible_bias),
        compute_visible_from_hidden=tf.nn.sigmoid(tf.matmul(rbm_in.y, tf.transpose(rbm_in.w)) + rbm_in.visible_bias))

def gr_init(rbm_in, config, sample_visible=False, sigma=1):
    hidden_p = tf.nn.sigmoid(tf.matmul(rbm_in.x, rbm_in.w) + rbm_in.hidden_bias)
    visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(rbm_in.w)) + rbm_in.visible_bias

    if sample_visible:
        visible_recon_p = sample_gaussian(visible_recon_p, sigma)

    hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, rbm_in.w) + rbm_in.hidden_bias)

    positive_grad = tf.matmul(tf.transpose(rbm_in.x), hidden_p)
    negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

    def f(x_old, x_new):
        return config.momentum * x_old +\
            config.learning_rate * x_new * (1 - config.momentum) / tf.to_float(tf.shape(x_new)[0])

    delta_w_new = f(rbm_in.delta_w, positive_grad - negative_grad)
    delta_visible_bias_new = f(rbm_in.delta_visible_bias, tf.reduce_mean(rbm_in.x - visible_recon_p, 0))
    delta_hidden_bias_new = f(rbm_in.delta_hidden_bias, tf.reduce_mean(hidden_p - hidden_recon_p, 0))

    update_delta_w = rbm_in.delta_w.assign(delta_w_new)
    update_delta_visible_bias = rbm_in.delta_visible_bias.assign(delta_visible_bias_new)
    update_delta_hidden_bias = rbm_in.delta_hidden_bias.assign(delta_hidden_bias_new)

    update_w = rbm_in.w.assign(rbm_in.w + delta_w_new)
    update_visible_bias = rbm_in.visible_bias.assign(rbm_in.visible_bias + delta_visible_bias_new)
    update_hidden_bias = rbm_in.hidden_bias.assign(rbm_in.hidden_bias + delta_hidden_bias_new)

    compute_hidden = tf.nn.sigmoid(tf.matmul(rbm_in.x, rbm_in.w) + rbm_in.hidden_bias)
    return RBMOutput(
        update_deltas=[update_delta_w, update_delta_visible_bias, update_delta_hidden_bias],
        update_weights=[update_w, update_visible_bias, update_hidden_bias],
        compute_hidden=compute_hidden,
        compute_visible=tf.matmul(compute_hidden, tf.transpose(rbm_in.w)) + rbm_in.visible_bias,
        compute_visible_from_hidden=tf.matmul(rbm_in.y, tf.transpose(rbm_in.w)) + rbm_in.visible_bias)

class RBM:
    def __init__(self, config, initialize):
        assert config is not None

        if not 0.0 <= config.momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        self.config = config
        self.input = RBMInput(config)
        self.output = initialize(self.input, self.config)

        self.compute_err = config.err_function(self.input.x, self.output.compute_visible)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def get_err(self, batch_x):
        return self.sess.run(self.compute_err, feed_dict={self.input.x: batch_x})

    def get_free_energy(self):
        pass

    def transform(self, batch_x):
        return self.sess.run(self.output.compute_hidden, feed_dict={self.input.x: batch_x})

    def transform_inv(self, batch_y):
        return self.sess.run(self.output.compute_visible_from_hidden, feed_dict={self.input.y: batch_y})

    def reconstruct(self, batch_x):
        return self.sess.run(self.output.compute_visible, feed_dict={self.input.x: batch_x})

    def partial_fit(self, batch_x):
        self.sess.run(self.output.update_weights + self.output.update_deltas, feed_dict={self.input.x: batch_x})

    def fit(self,
            data_x,
            n_epoches=10,
            batch_size=10,
            shuffle=True,
            verbose=True):
        assert n_epoches > 0

        n_data = data_x.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
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
                batch_x = data_x_cpy[b * batch_size:(b + 1) * batch_size]
                self.partial_fit(batch_x)
                batch_err = self.get_err(batch_x)
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

    def get_weights(self):
        return self.sess.run(self.input.w),\
            self.sess.run(self.input.visible_bias),\
            self.sess.run(self.input.hidden_bias)

    def save_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.input.w,
                                name + '_v': self.input.visible_bias,
                                name + '_h': self.input.hidden_bias})
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        self.sess.run(self.input.w.assign(w))
        self.sess.run(self.input.visible_bias.assign(visible_bias))
        self.sess.run(self.input.hidden_bias.assign(hidden_bias))

    def load_weights(self, filename, name):
        saver = tf.train.Saver({name + '_w': self.input.w,
                                name + '_v': self.input.visible_bias,
                                name + '_h': self.input.hidden_bias})
        saver.restore(self.sess, filename)

# ==================== demo =======================
from tqdm import tqdm
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images

config = RBMConfig(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, tqdm=tqdm)
bbrbm = RBM(config=config, initialize=bb_init)
errs = bbrbm.fit(mnist_images, n_epoches=30, batch_size=10)
plt.plot(errs)
plt.show()

IMAGE = 1

def show_digit(x):
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()

image = mnist_images[IMAGE]
image_rec = bbrbm.reconstruct(image.reshape(1,-1))

show_digit(image)
show_digit(image_rec)
