import time
import math
import numpy as np

def batch_generate(n, batchsize):
    total = n * batchsize
    return np.random.rand(total)

def avgevry2(indata):
    return (indata[0::2] + indata[1::2]) / 2

def base_name(var):
    """Extracts value passed to name= when creating a variable"""
    return var.name.split('/')[-1].split(':')[0]

def tc_mlp(matrix_dim):
    import tenncor as tc
    n_in = matrix_dim
    n_out = int(n_in / 2)
    batch_size = 1

    # regular mlp
    brain = tc.api.layer.link([
        tc.api.layer.dense([n_in], [matrix_dim]),
        tc.api.layer.bind(tc.api.sigmoid),
        tc.api.layer.dense([matrix_dim], [n_out]),
        tc.api.layer.bind(tc.api.sigmoid),
    ])

    invar = tc.Variable(np.zeros([batch_size, n_in], dtype=float), 'in')
    out = brain.connect(invar)
    expected_out = tc.Variable(np.zeros([batch_size, n_out], dtype=float), 'expected_out')
    err = tc.api.square(expected_out - out)

    test_batch = batch_generate(n_in, batch_size)
    test_batch_out = avgevry2(test_batch)
    test_batch = test_batch.reshape([batch_size, n_in])
    test_batch_out = test_batch_out.reshape([batch_size, n_out])

    start = time.time()
    invar.assign(test_batch)
    expected_out.assign(test_batch_out)
    print(err.release_get())
    dur = time.time() - start
    return dur

def tf_mlp(matrix_dim):
    import tensorflow as tf
    class Layer(object):
        def __init__(self, input_sizes, output_size, scope):
            """Cretes a neural network layer."""
            if type(input_sizes) != list:
                input_sizes = [input_sizes]

            self.input_sizes = input_sizes
            self.output_size = output_size
            self.scope       = scope or "Layer"

            with tf.compat.v1.variable_scope(self.scope):
                self.Ws = []
                for input_idx, input_size in enumerate(input_sizes):
                    W_name = "W_%d" % (input_idx,)
                    W_initializer =  tf.random_uniform_initializer(
                            -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
                    W_var = tf.compat.v1.get_variable(W_name, (input_size, output_size), initializer=W_initializer)
                    self.Ws.append(W_var)
                self.b = tf.compat.v1.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

        def __call__(self, xs):
            if type(xs) != list:
                xs = [xs]
            assert len(xs) == len(self.Ws), \
                    "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
            with tf.compat.v1.variable_scope(self.scope):
                return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

        def variables(self):
            return [self.b] + self.Ws

        def copy(self, scope=None):
            scope = scope or self.scope + "_copy"

            with tf.compat.v1.variable_scope(scope) as sc:
                for v in self.variables():
                    tf.compat.v1.get_variable(base_name(v), v.get_shape(),
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

            with tf.compat.v1.variable_scope(self.scope):
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
            with tf.compat.v1.variable_scope(self.scope):
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

    n_in = matrix_dim
    n_out = int(n_in / 2)
    batch_size = 1

    with tf.compat.v1.Session() as sess:
        brain = MLP([n_in], [matrix_dim, n_out], [tf.sigmoid, tf.sigmoid], scope='brain_' + str(matrix_dim))

        invar = tf.compat.v1.placeholder(tf.float32, [batch_size, n_in], name='invar')
        out = brain(invar)
        expected_out = tf.compat.v1.placeholder(tf.float32, [batch_size, n_out], name='expected_out')
        err = tf.square(expected_out - out)

        sess.run(tf.compat.v1.global_variables_initializer())

        test_batch = batch_generate(n_in, batch_size)
        test_batch_out = avgevry2(test_batch)
        test_batch = test_batch.reshape([batch_size, n_in])
        test_batch_out = test_batch_out.reshape([batch_size, n_out])

        start = time.time()
        print(sess.run(err, {
            invar: test_batch,
            expected_out: test_batch_out
        }))
        dur = time.time() - start
    return dur
