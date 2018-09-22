''' Generate test cases with tensorflow operations '''

import math
import tensorflow as tf
import numpy as np
import functools

import retrop.generate as gen
import retrop.client as client

N_LIMIT = 32000
RANK_LIMIT = 8

def make_shape(io, nrank=RANK_LIMIT):
    rank = io.get_arr("rank", int, 1, (1, nrank))[0]
    dimlimit = 10 ** (math.log(N_LIMIT, 10) / rank)
    shape = io.get_arr("shape", int, rank, (1, int(dimlimit)))
    n = functools.reduce(lambda d1, d2 : d1 * d2, shape)
    return shape, n, dimlimit

def unary(name, func, pos=False):
    io = gen.GenIO(name)
    shape, n, _ = make_shape(io)
    if pos:
        data = io.get_arr("data", float, n, (0.1, 1542.973))
    else:
        data = io.get_arr("data", float, n, (-1542.973, 1542.973))
    shaped_data = np.reshape(data, shape)

    var = tf.Variable(shaped_data)
    out = func(var)
    with tf.Session() as sess:
        sess.run(var.initializer)
        outdata = sess.run(out)
        io.set_arr("unary_out", list(np.reshape(outdata, [n])), float)
        io.send()

def binary(name, func, apos=False, bpos=False):
    io = gen.GenIO(name)
    shape, n, _ = make_shape(io)
    if apos:
        data = io.get_arr("data", float, n, (0.1, 1542.973))
    else:
        data = io.get_arr("data", float, n, (-1542.973, 1542.973))
    if bpos:
        data2 = io.get_arr("data2", float, n, (0.1, 1542.973))
    else:
        data2 = io.get_arr("data2", float, n, (-1542.973, 1542.973))
    shaped_data = np.reshape(data, shape)
    shaped_data2 = np.reshape(data2, shape)

    var = tf.Variable(shaped_data)
    var2 = tf.Variable(shaped_data2)
    out = func(var, var2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outdata = sess.run(out)
        io.set_arr("binary_out", list(np.reshape(outdata, [n])), float)
        io.send()

def matmul():
    io = gen.GenIO("tf_matmul")
    dimlimit = math.sqrt(N_LIMIT)
    ashape = io.get_arr("ashape", int, 2, (1, dimlimit))
    bdim = io.get_arr("bdim", int, 1, (1, dimlimit))[0]
    bshape = [bdim, ashape[0]]
    an = ashape[0] * ashape[1]
    bn = bdim * ashape[0]
    data = io.get_arr("data", float, an, (-1542.973, 1542.973))
    data2 = io.get_arr("data2", float, bn, (-1542.973, 1542.973))
    shaped_data = np.reshape(data, ashape[::-1])
    shaped_data2 = np.reshape(data2, bshape[::-1])

    var = tf.Variable(shaped_data)
    var2 = tf.Variable(shaped_data2)
    out = tf.matmul(var, var2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        outdata = sess.run(out)
        nout = outdata.shape[0] * outdata.shape[1]
        io.set_arr("matmul_out", list(np.reshape(outdata, [nout])), float)
        io.send()

if __name__ == "__main__":
    client.init("0.0.0.0:8581")
    print("client initialized")

    ufs = {
        'tf_abs': tf.abs,
        'tf_neg': tf.neg,
        'tf_sin': tf.sin,
        'tf_cos': tf.cos,
        'tf_tan': tf.tan,
        'tf_exp': tf.exp,
    }

    for name in ufs:
        print("generating " + name)
        unary(name, ufs[name])

    ufs_pos = {
        'tf_log': tf.log,
        'tf_sqrt': tf.sqrt,
    }

    for name in ufs_pos:
        print("generating " + name)
        unary(name, ufs_pos[name], pos=True)

    bfs = {
        'tf_pow': tf.pow,
        "tf_add": tf.add,
        "tf_sub": tf.sub,
        "tf_mul": tf.mul,
    }

    for name in bfs:
        print("generating " + name)
        binary(name, bfs[name])

    print("generating tf_div")
    binary("tf_div", tf.div, bpos=True)
    print("generating tf_matmul")
    matmul()
