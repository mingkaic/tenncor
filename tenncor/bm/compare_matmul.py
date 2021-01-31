import time
import numpy as np

def np_matmul(matrix_dim):
    shape = [matrix_dim, matrix_dim]
    data = np.random.rand(*shape)
    data2 = np.random.rand(*shape)

    start = time.time()
    print(data.dot(data2))
    dur = time.time() - start
    return dur

def tc_matmul(matrix_dim):
    shape = [matrix_dim, matrix_dim]
    import tenncor as tc

    data = np.random.rand(*shape)
    data2 = np.random.rand(*shape)

    var = tc.Variable(data, 'var')
    var2 = tc.Variable(data2, 'var2')
    out = tc.api.matmul(var, var2)

    start = time.time()
    print(out.release_get())
    dur = time.time() - start
    return dur

def tf_matmul(matrix_dim):
    shape = [matrix_dim, matrix_dim]
    import tensorflow as tf

    data = np.random.rand(*shape)
    data2 = np.random.rand(*shape)

    dur = 10000
    with tf.compat.v1.Session() as sess:
        var = tf.Variable(data)
        var2 = tf.Variable(data2)
        out = tf.matmul(var, var2)

        sess.run(var.initializer)
        sess.run(var2.initializer)

        start = time.time()
        print(sess.run(out))
        dur = time.time() - start
    return dur
