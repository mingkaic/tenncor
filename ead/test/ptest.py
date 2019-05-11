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
        data = np.random.rand(*shape)
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

    def test_convolution(self):
        padding = "VALID"
        shapes = [
            ([3, 3], [3, 3]),
            ([5, 5], [3, 3]),
        ]
        for shape, kernelshape in shapes:
            tf_shape = [1, shape[0], shape[1], 1]
            tf_kernelshape = [kernelshape[0], kernelshape[1], 1, 1]

            data = np.random.rand(*shape).astype(np.float32)
            kernel = np.random.rand(*kernelshape).astype(np.float32)
            tf_data = data.reshape(tf_shape)
            tf_kdata = kernel.reshape(tf_kernelshape)

            var = ead.variable(data, 'var')
            vkernel = ead.variable(kernel, 'vkernel')

            tf_var = tf.Variable(tf_data)
            tf_kernel = tf.Variable(tf_kdata)

            out = age.convolution(var, vkernel, list(range(8)))

            sess = ead.Session()
            sess.track(out)
            sess.update()

            fout = out.get()

            tfsess = tf.Session()
            tfsess.run(tf_var.initializer)
            tfsess.run(tf_kernel.initializer)

            tf_out = tf.nn.convolution(tf_var, tf_kernel, padding)
            tf_fout = tfsess.run(tf_out)

            tf_fout = tf_fout.reshape([tf_fout.shape[1], tf_fout.shape[2]])
            self._array_close(tf_fout, fout)

            var2 = ead.variable(data, 'var2')
            zero = ead.derive(out, var2)
            ex = ead.derive(out, var)
            ex2 = ead.derive(out, vkernel)

            sess.track(zero)
            sess.track(ex)
            sess.track(ex2)
            sess.update()

            rej = zero.get()
            der = ex.get()
            der2 = ex2.get()

            data0 = np.zeros(shape, dtype=np.float32)
            tf_grad, tf_grad2 = tf.gradients(tf_out, [tf_var, tf_kernel])

            exdata = tfsess.run(tf_grad)
            exdata2 = tfsess.run(tf_grad2)

            exdata = exdata.reshape(shape)
            exdata2 = exdata2.reshape(kernelshape)

            self._array_eq(data0, rej)
            self._array_close(exdata, der)
            self._array_close(exdata2, der2)

    def test_grader_scenario1(self): # REDUCE -> MUL
        data = np.random.rand(3,10)
        data2 = np.random.rand(10)

        var = ead.variable(data, 'var')
        var2 = ead.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = age.mul(age.reduce_sum(var, 1), var2)
        tf_out = tf.multiply(tf.reduce_sum(tf_var, 0), tf_var2)

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = ead.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario2(self): # EXTEND -> MUL
        data = np.random.rand(10)
        data2 = np.random.rand(3,10)

        var = ead.variable(data, 'var')
        var2 = ead.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = age.mul(age.extend(var, 1, [3]), var2)
        tf_out = tf_var * tf_var2

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = ead.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario3(self): # PERMUTE -> MUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,10)

        var = ead.variable(data, 'var')
        var2 = ead.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = age.mul(age.permute(var, [1,0]), var2)
        tf_out = tf.transpose(tf_var) * tf_var2

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = ead.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario4(self): # MATMUL -> MUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,5)
        data3 = np.random.rand(10,5)

        var = ead.variable(data, 'var')
        var2 = ead.variable(data2, 'var2')
        var3 = ead.variable(data3, 'var3')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_var3 = tf.Variable(data3)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)
        tfsess.run(tf_var3.initializer)

        out = age.mul(age.matmul(var, var2), var3)
        tf_out = tf.multiply(tf.matmul(tf_var, tf_var2), tf_var3)

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = ead.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario5(self): # MATMUL -> MATMUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,5)
        data3 = np.random.rand(5,4)

        var = ead.variable(data, 'var')
        var2 = ead.variable(data2, 'var2')
        var3 = ead.variable(data3, 'var3')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_var3 = tf.Variable(data3)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)
        tfsess.run(tf_var3.initializer)

        out = age.matmul(age.matmul(var, var2), var3)
        tf_out = tf.matmul(tf.matmul(tf_var, tf_var2), tf_var3)

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = ead.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario6(self): # REDUCE -> MATMUL
        data = np.random.rand(4,10,3)
        data2 = np.random.rand(3,5)

        var = ead.variable(data, 'var')
        var2 = ead.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = age.matmul(age.reduce_sum(var, 2), var2)
        tf_out = tf.matmul(tf.reduce_sum(tf_var, 0), tf_var2)

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = ead.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario7(self): # EXTEND -> MATMUL
        data = np.random.rand(3)
        data2 = np.random.rand(3,5)
        ones = np.ones([10, 3])

        var = ead.variable(data, 'var')
        var2 = ead.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = age.matmul(age.extend(var, 1, [10]), var2)
        tf_out = tf.matmul(tf_var * ones, tf_var2)

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = ead.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

	# A<a> -> EXTEND -> B<a,b>
	# A<a> -> EXTEND -> C<a,c> -> PERMUTE -> <c,a>
	# B MATMUL C -> D<c,b>
    def test_grader_scenario8(self):
        data = np.random.rand(5)
        ones = np.ones([10, 5])
        ones2 = np.ones([3, 5])

        var = ead.variable(data, 'var')
        tf_var = tf.Variable(data)

        tfsess = tf.Session()
        tfsess.run(tf_var.initializer)

        out = age.matmul(
            age.extend(var, 1, [10]),
            age.permute(age.extend(var, 1, [3]), [1, 0]))
        tf_out = tf.matmul(tf_var * ones, tf.transpose(tf_var * ones2))

        # evaluate regular matmul
        sess = ead.Session()
        sess.track(out)
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = ead.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track(ex)
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario9(self):
        ashape = [2, 3]
        bshape = [3, 4]
        cshape = [4, 2]

        data = np.random.rand(*ashape)
        data2 = np.random.rand(*bshape)
        data3 = np.random.rand(*cshape)

        a = ead.variable(data, 'a')
        b = ead.variable(data2, 'b')
        c = ead.variable(data3, 'c')
        tf_a = tf.Variable(data)
        tf_b = tf.Variable(data2)
        tf_c = tf.Variable(data3)

        d = age.matmul(a, b)
        e = age.matmul(c, d)
        f = age.matmul(age.transpose(d), age.transpose(c))
        dest = age.matmul(e, f)

        tf_d = tf.matmul(tf_a, tf_b)
        tf_e = tf.matmul(tf_c, tf_d)
        tf_f = tf.matmul(tf.transpose(tf_d), tf.transpose(tf_c))
        tf_dest = tf.matmul(tf_e, tf_f)

        da = ead.derive(dest, a)
        db = ead.derive(dest, b)
        dc = ead.derive(dest, c)
        tf_da, tf_db, tf_dc = tf.gradients(tf_dest, [tf_a, tf_b, tf_c])

        tfsess = tf.Session()
        tfsess.run(tf_a.initializer)
        tfsess.run(tf_b.initializer)
        tfsess.run(tf_c.initializer)

        sess = ead.Session()
        sess.track(dest)
        sess.track(da)
        sess.track(db)
        sess.track(dc)
        sess.update()

        exa = tfsess.run(tf_da)
        exb = tfsess.run(tf_db)
        exc = tfsess.run(tf_dc)
        self._array_close(exa, da.get())
        self._array_close(exb, db.get())
        self._array_close(exc, dc.get())

if __name__ == "__main__":
    unittest.main()
