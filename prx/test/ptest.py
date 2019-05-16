import unittest
import numpy as np
import tensorflow as tf

import ead.ead as ead
import prx.prx as prx

def _normalize_shape(arr1, arr2):
    if 'shape' in dir(arr1):
        shape1 = arr1.shape
        if not isinstance(shape1, tuple):
            shape1 = []
    else:
        shape1 = []
    if 'shape' in dir(arr2):
        shape2 = arr2.shape
        if not isinstance(shape2, tuple):
            shape2 = []
    else:
        shape2 = []

    n1 = len(shape1)
    n2 = len(shape2)
    i = 0
    while i < n1 and shape1[i] == 1:
        i = i + 1
    shape1 = shape1[i:]
    i = 0
    while i < n2 and shape2[i] == 1:
        i = i + 1
    shape2 = shape2[i:]

    n1 = len(shape1)
    n2 = len(shape2)
    maxn = max(n1, n2)
    normalized_s1 = list(shape1) + [1] * (maxn - n1)
    normalized_s2 = list(shape2) + [1] * (maxn - n2)
    return normalized_s1, normalized_s2

class PRXTest(unittest.TestCase):
    def _array_eq(self, arr1, arr2):
        msg = 'diff arrays:\n{}\n{}'.format(arr1, arr2)
        s1, s2 = _normalize_shape(arr1, arr2)
        if 'shape' in dir(arr1):
            arr1 = arr1.reshape(s1)
        else:
            arr1 = np.array(arr1).reshape(s1)
        if 'shape' in dir(arr2):
            arr2 = arr2.reshape(s2)
        else:
            arr2 = np.array(arr2).reshape(s2)
        self.assertTrue(np.array_equal(arr1, arr2), msg)

    def _array_close(self, arr1, arr2):
        def prod(arr):
            return reduce(lambda acc, s: acc * s, arr + [1])
        msg = 'vastly diff arrays:\n{}\n{}'.format(arr1, arr2)
        if isinstance(arr1, int):
            arr1 = np.array([arr1])
        if isinstance(arr2, int):
            arr2 = np.array([arr2])
        avoidshape = 1 == prod(list(arr1.shape)) and\
            1 == prod(list(arr2.shape))
        s1, s2 = _normalize_shape(arr1, arr2)
        self.assertTrue(np.allclose(arr1, arr2, atol=1e-05) and s1 == s2, msg)

    def test_conv2d(self):
        # batch, height, width, in
        shape = [3, 3, 3, 2]
        data = np.random.rand(*shape).astype(np.float32)

        # height, width, in, out
        kshape = [2, 2, 2, 4]
        kdata = np.random.rand(*kshape).astype(np.float32)

        image = ead.variable(data, 'image')
        kernel = ead.variable(kdata, 'vkernel')
        zero = ead.scalar_constant(0, [4])

        tfimage = tf.Variable(data)
        tfkernel = tf.Variable(kdata)

        out = prx.conv2d(image, kernel, zero)

        sess = ead.Session()
        sess.track(out)
        sess.update()

        tfsess = tf.Session()

        tfoutput = tf.nn.conv2d(tfimage, tfkernel, [1, 1, 1, 1], 'VALID')
        tfsess.run(tfimage.initializer)
        tfsess.run(tfkernel.initializer)

        conv_output = out.get()
        tfconv_output = tfsess.run(tfoutput)
        self._array_close(tfconv_output, conv_output)

        var2 = ead.variable(data, 'var2')
        zero = ead.derive(out, var2)
        ex = ead.derive(out, image)
        ex2 = ead.derive(out, kernel)

        sess.track(zero)
        sess.track(ex)
        sess.track(ex2)
        sess.update()

        rej = zero.get()
        der = ex.get()
        der2 = ex2.get()

        data0 = np.zeros(shape, dtype=np.float32)
        tf_grad, tf_grad2 = tf.gradients(tfoutput, [tfimage, tfkernel])

        exdata = tfsess.run(tf_grad)
        exdata2 = tfsess.run(tf_grad2)

        self._array_eq(data0, rej)
        self._array_close(exdata, der)
        self._array_close(exdata2, der2)

if __name__ == "__main__":
    unittest.main()
