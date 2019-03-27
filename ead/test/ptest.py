import unittest
import numpy as np
import tensorflow as tf

import ead.age as age
import ead.ead as ead

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

class EADTest(unittest.TestCase):
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

    def _common_unary(self, shape, api, real, derive):
        data = np.random.rand(*shape) * 34
        var = ead.variable(data, 'var')
        out = api(var)

        sess = ead.Session()
        sess.track(out)
        sess.update()

        fout = out.get()
        self._array_close(real(data), fout)

        var2 = ead.variable(data, 'var2')
        ex = ead.derive(out, var)
        zero = ead.derive(out, var2)

        sess.track(ex)
        sess.track(zero)
        sess.update()

        data0 = np.zeros(shape, dtype=np.float32)
        der = ex.get()
        rej = zero.get()
        exdata = derive(data)
        self._array_close(exdata, der)
        self._array_eq(data0, rej)

    def _common_unary_tf(self, shape, api, tf_op):
        data = np.random.rand(*shape)
        var = ead.variable(data, 'var')
        out = api(var)

        tf_var = tf.Variable(data)
        tf_out = tf_op(tf_var)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)

        sess = ead.Session()
        sess.track(out)
        sess.update()

        fout = out.get()
        real = tfsess.run(tf_out)
        self._array_close(real, fout)

        var2 = ead.variable(data, 'var2')
        ex = ead.derive(out, var)
        zero = ead.derive(out, var2)

        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.track(zero)
        sess.update()

        data0 = np.zeros(shape, dtype=np.float32)
        der = ex.get()
        rej = zero.get()
        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, der)
        self._array_eq(data0, rej)

    def _common_binary(self, shape, api, real, derive):
        data = np.random.rand(*shape)
        data2 = np.random.rand(*shape)
        var = ead.variable(data, 'var')
        var2 = ead.variable(data2, 'var2')
        out = api(var, var2)
        both = api(var, var)

        sess = ead.Session()
        sess.track(out)
        sess.track(both)
        sess.update()

        fout = out.get()
        fboth = both.get()
        self._array_close(real(data, data2), fout)
        self._array_close(real(data, data), fboth)

        var3 = ead.variable(data, 'var3')

        zero = ead.derive(out, var3)
        ex = ead.derive(out, var)
        ex2 = ead.derive(out, var2)
        ex3 = ead.derive(both, var)
        sess.track(zero)
        sess.track(ex)
        sess.track(ex2)
        sess.track(ex3)
        sess.update()

        rej = zero.get()
        der = ex.get()
        der2 = ex2.get()
        der3 = ex3.get()

        data0 = np.zeros(shape, dtype=np.float32)
        exdata = derive(0, (data, data2))
        exdata2 = derive(1, (data, data2))
        exdata3 = derive(0, (data, data)) + derive(1, (data, data))

        self._array_eq(data0, rej)
        self._array_close(exdata, der)
        self._array_close(exdata2, der2)
        self._array_close(exdata3, der3)

    def _common_reduce_1d(self, dim_reduce, tf_reduce):
        shape = [3, 4, 5]
        data = np.random.rand(*shape)
        var = ead.variable(data, 'var')
        tf_var = tf.Variable(data)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)

        out = dim_reduce(var, 1)
        tf_out = tf_reduce(tf_var, [1])

        sess = ead.Session()
        sess.track(out)
        sess.update()

        fout = out.get()
        tf_fout = tfsess.run(tf_out)

        self._array_close(tf_fout, fout)

        var2 = ead.variable(data, 'var2')
        ex = ead.derive(out, var)
        zero = ead.derive(out, var2)
        sess.track(ex)
        sess.track(zero)
        sess.update()

        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        data0 = np.zeros(shape, dtype=np.float32)
        der = ex.get()
        rej = zero.get()

        exdata = tfsess.run(tf_grad)

        self._array_close(exdata, der)
        self._array_eq(data0, rej)

    def _common_reduce(self, all_reduce, dim_reduce, tf_reduce):
        shape = [3, 4, 5]
        # data = np.random.rand(*shape)
        data = np.array([
            [[ 1,  2,  3,  4,  5], [ 6,  7,  8,  9, 10], [11, 12, 21, 22, 23], [24, 25, 26, 27, 28]],
            [[29, 20, 31, 32, 31], [32, 33, 34, 35, 36], [37, 38, 39, 30, 41], [42, 41, 42, 43, 44]],
            [[45, 46, 47, 48, 49], [40, 51, 52, 51, 52], [53, 54, 55, 56, 57], [58, 59, 50, 61, 62]]
        ], dtype=np.float32)
        var = ead.variable(data, 'var')
        tf_var = tf.Variable(data)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)

        out = all_reduce(var)
        out2 = dim_reduce(var, 1)
        tf_out = tf_reduce(tf_var)
        tf_out2 = tf_reduce(tf_var, [0, 1])

        sess = ead.Session()
        sess.track(out)
        sess.track(out2)
        sess.update()

        fout = out.get()
        fout2 = out2.get()
        tf_fout = np.array(tfsess.run(tf_out))
        tf_fout2 = tfsess.run(tf_out2)

        self._array_close(tf_fout, fout)
        self._array_close(tf_fout2, fout2)

        var2 = ead.variable(data, 'var2')
        ex = ead.derive(out, var)
        ex2 = ead.derive(out2, var)
        zero = ead.derive(out, var2)
        sess.track(ex)
        sess.track(ex2)
        sess.track(zero)
        sess.update()

        tf_grad = tf.gradients(tf_out, [tf_var])[0]
        tf_grad2 = tf.gradients(tf_out2, [tf_var])[0]

        data0 = np.zeros(shape, dtype=np.float32)
        der = ex.get()
        der2 = ex2.get()
        rej = zero.get()

        exdata = tfsess.run(tf_grad)
        exdata2 = tfsess.run(tf_grad2)

        self._array_close(exdata, der)
        self._array_close(exdata2, der2)
        self._array_eq(data0, rej)

    def test_variable(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=np.float32)
        data0 = np.zeros(shape, dtype=np.float32)
        data = np.random.rand(3, 4, 5) * 234
        var = ead.variable(data, 'var')

        sess = ead.Session()
        sess.track(var)
        sess.update()
        fout = var.get()

        self.assertEqual(tuple(shape), fout.shape)
        self._array_close(data, fout)

        var2 = ead.variable(data, 'var2')
        one = ead.derive(var, var)
        zero = ead.derive(var, var2)
        sess.track(one)
        sess.track(zero)
        sess.update()

        out1 = one.get()
        out0 = zero.get()
        self.assertEqual(tuple(shape), out1.shape)
        self.assertEqual(tuple(shape), out0.shape)
        self._array_eq(data1, out1)
        self._array_eq(data0, out0)

    def test_abs(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.abs, abs,
            lambda data: data / abs(data))

    def test_neg(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=np.float32)
        self._common_unary(shape, age.neg, lambda a: -a,
            lambda data: -data1)

    def test_sin(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.sin, np.sin, np.cos)

    def test_cos(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.cos, np.cos, lambda x: -np.sin(x))

    def test_tan(self):
        shape = [3, 4, 5]
        self._common_unary_tf(shape, age.tan, tf.tan)

    def test_exp(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.exp, np.exp, np.exp)

    def test_log(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.log, np.log, lambda x: 1.0 / x)

    def test_sqrt(self):
        shape = [3, 4, 5]
        self._common_unary(shape, age.sqrt, np.sqrt,
            lambda x: 1.0 / (2.0 * np.sqrt(x)))

    def test_round(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=np.float32)
        self._common_unary(shape, age.round, np.round, lambda x: data1)

    def test_sigmoid(self):
        shape = [3, 4, 5]
        self._common_unary_tf(shape, age.sigmoid, tf.sigmoid)

    def test_tanh(self):
        shape = [3, 4, 5]
        self._common_unary_tf(shape, age.tanh, tf.tanh)

    def test_square(self):
        shape = [3, 4, 5]
        self._common_unary_tf(shape, age.square, tf.square)

    def test_cube(self):
        shape = [3, 4, 5]
        self._common_unary_tf(shape, age.cube, lambda x: tf.pow(x, 3))

    def test_pow(self):
        shape = [3, 4, 5]
        def pow_der(i, data):
            a, b = data
            if i == 0:
                return b * a ** (b - 1)
            return a ** b * np.log(a)
        self._common_binary(shape, age.pow, lambda x, y: x ** y, pow_der)

    def test_add(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=np.float32)
        self._common_binary(shape, age.add, lambda x, y: x + y,
            lambda i, data: data1)

    def test_sub(self):
        shape = [3, 4, 5]
        data1 = np.ones(shape, dtype=np.float32)
        def sub_der(i, data):
            if i == 0:
                return data1
            return -data1
        self._common_binary(shape, age.sub, lambda x, y: x - y, sub_der)

    def test_mul(self):
        shape = [3, 4, 5]
        def mul_der(i, data):
            if i == 0:
                return data[1]
            return data[0]
        self._common_binary(shape, age.mul, lambda x, y: x * y, mul_der)

    def test_div(self):
        shape = [3, 4, 5]
        def div_der(i, data):
            a, b = data
            if i == 0:
                return 1 / b
            return -a / (b * b)
        self._common_binary(shape, age.div, lambda x, y: x / y, div_der)

    def test_min(self):
        shape = [3, 4, 5]
        def min_der(i, data):
            a, b = data
            if i == 0:
                return (a <= b).astype(float)
            return (b <= a).astype(float)
        self._common_binary(shape, age.min, np.minimum, min_der)

    def test_max(self):
        shape = [3, 4, 5]
        def max_der(i, data):
            a, b = data
            if i == 0:
                return (a >= b).astype(float)
            return (b >= a).astype(float)
        self._common_binary(shape, age.max, np.maximum, max_der)

    def test_eq(self):
        shape = [3, 4, 5]
        data0 = np.zeros(shape, dtype=np.float32)
        self._common_binary(shape,
            lambda x, y: age.eq(age.round(x), age.round(y)),
            lambda x, y: np.round(x) == np.round(y),
            lambda i, data: data0)

    def test_neq(self):
        shape = [3, 4, 5]
        data0 = np.zeros(shape, dtype=np.float32)
        self._common_binary(shape,
            lambda x, y: age.neq(age.round(x), age.round(y)),
            lambda x, y: np.round(x) != np.round(y),
            lambda i, data: data0)

    def test_lt(self):
        shape = [3, 4, 5]
        data0 = np.zeros(shape, dtype=np.float32)
        self._common_binary(shape,
            lambda x, y: age.lt(age.round(x), age.round(y)),
            lambda x, y: np.round(x) < np.round(y),
            lambda i, data: data0)

    def test_gt(self):
        shape = [3, 4, 5]
        data0 = np.zeros(shape, dtype=np.float32)
        self._common_binary(shape,
            lambda x, y: age.gt(age.round(x), age.round(y)),
            lambda x, y: np.round(x) > np.round(y),
            lambda i, data: data0)

    def test_nelems(self):
        shape = [3, 4, 5]
        data0 = np.zeros(shape, dtype=np.float32)
        self._common_unary(shape, age.n_elems,
            lambda data: np.prod(data.shape),
            lambda data: data0)

    def test_ndims(self):
        shape = [3, 4, 5]
        data0 = np.zeros(shape, dtype=np.float32)
        self._common_unary(shape,
            lambda x: age.n_dims(x, 0),
            lambda data: data.shape[2],
            lambda data: data0)

    def test_extend(self):
        shape = [2]
        data = np.random.rand(*shape) * 13
        expected_out = np.array(list(data) * 3).reshape([3, 2])
        var = ead.variable(data, 'var')

        out = age.extend(var, 1, [3])
        sess = ead.Session()
        sess.track(out)
        sess.update()

        fout = out.get()
        self._array_close(expected_out, fout)

        ex = ead.derive(out, var)
        sess.track(ex)
        sess.update()

        der = ex.get()
        self._array_close(np.array([3, 3]), der)

    def test_rsum_1d(self):
        self._common_reduce_1d(age.reduce_sum_1d, tf.reduce_sum)

    def test_rprod_1d(self):
        self._common_reduce_1d(age.reduce_prod_1d, tf.reduce_prod)

    def test_rmin_1d(self):
        self._common_reduce_1d(age.reduce_min_1d, tf.reduce_min)

    def test_rmax_1d(self):
        self._common_reduce_1d(age.reduce_max_1d, tf.reduce_max)

    def test_rsum(self):
        self._common_reduce(age.reduce_sum, age.reduce_sum, tf.reduce_sum)

    def test_rprod(self):
        self._common_reduce(age.reduce_prod, age.reduce_prod, tf.reduce_prod)

    def test_rmin(self):
        self._common_reduce(age.reduce_min, age.reduce_min, tf.reduce_min)

    def test_rmax(self):
        self._common_reduce(age.reduce_max, age.reduce_max, tf.reduce_max)

    def test_matmul(self):
        shape = [5, 5]
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
        both = age.matmul(var, var)

        # tensorflow matmul
        tf_out = tf.matmul(tf_var, tf_var2)
        tf_both = tf.matmul(tf_var, tf_var)

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.track(both)
        sess.update()

        fout = out.get()
        fboth = both.get()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)
        tf_fboth = tfsess.run(tf_both)

        # check regular matmul
        self._array_close(tf_fout, fout)
        self._array_close(tf_fboth, fboth)

        var3 = ead.variable(data, 'var3')

        zero = ead.derive(out, var3)
        ex = ead.derive(out, var)
        ex2 = ead.derive(out, var2)
        ex3 = ead.derive(both, var)

        sess.track(zero)
        sess.track(ex)
        sess.track(ex2)
        sess.track(ex3)
        sess.update()

        # eval derivative of regular matmul
        rej = zero.get()
        der = ex.get()
        der2 = ex2.get()
        der3 = ex3.get()

        # eval derivative of tensorflow matmul
        data0 = np.zeros(shape, dtype=np.float32)
        tf_grad, tf_grad2 = tf.gradients(tf_out, [tf_var, tf_var2])
        tf_grad3 = tf.gradients(tf_both, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        exdata2 = tfsess.run(tf_grad2)
        exdata3 = tfsess.run(tf_grad3)

        # check regular matmul
        self._array_eq(data0, rej)
        self._array_close(exdata, der)
        self._array_close(exdata2, der2)
        self._array_close(exdata3, der3)

    # def test_convolution(self):
    #     padding = "VALID"
    #     batchsize = 2
    #     inchannel = 3
    #     outchannel = 4
    #     dims = 5

    #     shapes = [
    #         ([1, 3, 3, 1], [3, 3, 1, 1]),
    #         ([batchsize, dims, dims, inchannel], [3, 3, inchannel, outchannel]),
    #     ]
    #     for shape, kernelshape in shapes:
    #         data = np.random.rand(*shape).astype(np.float32)
    #         kernel = np.random.rand(*kernelshape).astype(np.float32)

    #         var = ead.variable(data, 'var')
    #         vkernel = ead.variable(kernel, 'vkernel')
    #         tf_var = tf.Variable(data)
    #         tf_kernel = tf.Variable(kernel)

    #         sess = tf.Session()
    #         sess.run(tf_var.initializer)
    #         sess.run(tf_kernel.initializer)

    #         out = age.convolution(var, vkernel)
    #         tf_out = tf.nn.convolution(tf_var, tf_kernel, padding)

    #         fout = out.evaluate(dtype=np.dtype(float))
    #         tf_fout = sess.run(tf_out)

    #         self._array_close(tf_fout, fout)

    #         var2 = ead.variable(data, 'var2')
    #         zero = ead.derive(out, var2)
    #         ex = ead.derive(out, var)
    #         ex2 = ead.derive(out, vkernel)

    #         rej = zero.evaluate()
    #         der = ex.evaluate()
    #         der2 = ex2.evaluate()

    #         data0 = np.zeros(shape, dtype=np.float32)
    #         tf_grad, tf_grad2 = tf.gradients(tf_out, [tf_var, tf_kernel])

    #         exdata = sess.run(tf_grad)
    #         exdata2 = sess.run(tf_grad2)

    #         self._array_eq(data0, rej)
    #         self._array_close(exdata, der)
    #         self._array_close(exdata2, der2)

if __name__ == "__main__":
    unittest.main()
