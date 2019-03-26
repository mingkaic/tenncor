import sys
import time
import argparse
import math

import numpy as np

import tensorflow as tf

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

def base_name(var):
    """Extracts value passed to name= when creating a variable"""
    return var.name.split('/')[-1].split(':')[0]

class Layer(object):
    def __init__(self, input_sizes, output_size, scope):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]

        self.input_sizes = input_sizes
        self.output_size = output_size
        self.scope       = scope or "Layer"

        with tf.variable_scope(self.scope):
            self.Ws = []
            for input_idx, input_size in enumerate(input_sizes):
                W_name = "W_%d" % (input_idx,)
                W_initializer =  tf.random_uniform_initializer(
                        -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
                W_var = tf.get_variable(W_name, (input_size, output_size), initializer=W_initializer)
                self.Ws.append(W_var)
            self.b = tf.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws), \
                "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
        with tf.variable_scope(self.scope):
            return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

    def variables(self):
        return [self.b] + self.Ws

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"

        with tf.variable_scope(scope) as sc:
            for v in self.variables():
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32,partition_info=None: v.initialized_value())
            sc.reuse_variables()
            return Layer(self.input_sizes, self.output_size, scope=sc)

class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities, scope=None, given_layers=None):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.scope = scope or "MLP"

        assert len(hiddens) == len(nonlinearities), \
                "Number of hiddens must be equal to number of nonlinearities"

        with tf.variable_scope(self.scope):
            if given_layers is not None:
                self.input_layer = given_layers[0]
                self.layers      = given_layers[1:]
            else:
                self.input_layer = Layer(input_sizes, hiddens[0], scope="input_layer")
                self.layers = []

                for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
                    self.layers.append(Layer(h_from, h_to, scope="hidden_layer_%d" % (l_idx,)))

    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        with tf.variable_scope(self.scope):
            hidden = self.input_nonlinearity(self.input_layer(xs))
            for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
                hidden = nonlinearity(layer(hidden))
            return hidden

    def variables(self):
        res = self.input_layer.variables()
        for layer in self.layers:
            res.extend(layer.variables())
        return res

    def copy(self, scope=None):
        scope = scope or self.scope + "_copy"
        nonlinearities = [self.input_nonlinearity] + self.layer_nonlinearities
        given_layers = [self.input_layer.copy()] + [layer.copy() for layer in self.layers]
        return MLP(self.input_sizes, self.hiddens, nonlinearities, scope=scope,
                given_layers=given_layers)

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
    args = parser.parse_args(args)

    if args.seed:
        print('seeding {}'.format(args.seedval))
        np.random.seed(args.seedval)

    learning_rate = 0.9

    n_in = 10
    n_out = n_in / 2

    brain = MLP([n_in], [9, n_out], [tf.sigmoid, tf.sigmoid])

    sess = tf.Session()
    n_batch = args.n_batch
    show_every_n = 500

    trained_in = tf.placeholder(tf.float32, [None, n_in], name='trained_in')
    trained_out = brain(trained_in)
    expected_out = tf.placeholder(tf.float32, [n_batch, n_out], name='expected_out')

    error = tf.square(expected_out - trained_out)

    layer0 = brain.input_layer
    layer1 = brain.layers[0]

    w0 = layer0.Ws[0]
    b0 = layer0.b
    w1 = layer1.Ws[0]
    b1 = layer1.b

    dw0, db0, dw1, db1 = tf.gradients(error, [w0, b0, w1, b1])

    w0_update = w0.assign_sub(learning_rate * dw0)
    b0_update = b0.assign_sub(learning_rate * db0)
    w1_update = w1.assign_sub(learning_rate * dw1)
    b1_update = b1.assign_sub(learning_rate * db1)

    print(w0)

    def calculate_update(batch, batch_out):
        out, _, _, _, _ = sess.run([
            error,
            w0_update,
            b0_update,
            w1_update,
            b1_update,
        ], {
            trained_in: batch,
            expected_out: batch_out,
        })

        return out

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # gradients = optimizer.compute_gradients(error)
    # train_op = optimizer.apply_gradients(gradients)

    sess.run(tf.global_variables_initializer())
    start = time.time()
    for i in range(args.n_train):
        batch = batch_generate(n_in, n_batch)
        batch_out = avgevry2(batch)
        batch = batch.reshape([n_batch, n_in])
        batch_out = batch_out.reshape([n_batch, n_out])
        # trained_derr, _ = sess.run([
        #     error,
        #     train_op,
        # ], {
        #     trained_in: batch,
        #     expected_out: batch_out
        # })
        trained_derr = calculate_update(batch, batch_out)
        if i % show_every_n == show_every_n - 1:
            print('training {}\ntraining error:\n{}'
                .format(i + 1, trained_derr))

    print('training time: {} seconds'.format(time.time() - start))

    trained_err = 0
    for i in range(args.n_test):
        if i % show_every_n == show_every_n - 1:
            print('testing {}'.format(i + 1))

        test_batch = batch_generate(n_in, 1)
        test_batch_out = avgevry2(test_batch)
        test_batch = test_batch.reshape([1, n_in])
        test_batch_out = test_batch_out.reshape([1, n_out])

        trained_data = sess.run(trained_out, {
            trained_in: test_batch
        })

        trained_err += np.mean(abs(trained_data - test_batch_out))

    trained_err /= args.n_test
    print('trained mlp error rate: {}%'.format(trained_err * 100))

if '__main__' == __name__:
    main(sys.argv[1:])
