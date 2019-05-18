import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import ead.age as age
import ead.ead as ead

matrix_dims = [
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
]

np_durs = []
llo_durs = []
ead_durs = []
tf_durs = []
for matrix_dim in matrix_dims:
    shape = [matrix_dim, matrix_dim]
    data = np.random.rand(*shape)
    data2 = np.random.rand(*shape)

    var = ead.variable(data, 'var')
    var2 = ead.variable(data2, 'var2')
    tf_var = tf.Variable(data)
    tf_var2 = tf.Variable(data2)

    tfsess = tf.Session()
    tfsess.run(tf_var.initializer)
    tfsess.run(tf_var2.initializer)

    # regular matmul
    out = age.matmul(var, var2)

    # tensorflow matmul
    tf_out = tf.matmul(tf_var, tf_var2)

    # numpy matmul
    start = time.time()
    print(data.dot(data2))
    np_dur = time.time() - start

    sess = ead.Session()
    sess.track(out)

    start = time.time()
    sess.update()
    print(out.get())
    ead_dur = time.time() - start

    start = time.time()
    tf_fout = tfsess.run(tf_out)
    print(tf_fout)
    tf_dur = time.time() - start

    np_durs.append(np_dur)
    ead_durs.append(ead_dur)
    tf_durs.append(tf_dur)

print('numpy durations: ', np_durs)
print('ead durations: ', ead_durs)
print('tf durations: ', tf_durs)
ead_line = plt.plot(matrix_dims, ead_durs, 'r--', label='ead durations')
np_lines = plt.plot(matrix_dims, np_durs, 'g--', label='numpy durations')
tf_line = plt.plot(matrix_dims, tf_durs, 'b--', label='tf durations')
plt.legend()
plt.show()
