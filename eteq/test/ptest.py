import json
import random
import logging
import unittest
import numpy as np
import tensorflow as tf

import eteq.tenncor as tc
import eteq.eteq as eteq

from testutil.generate_testcases import generate_testcases

_test_data = {}

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

def _round_helper(x):
    if isinstance(x, float):
        return round(x)
    return tc.round(x)

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
        s1, s2 = _normalize_shape(arr1, arr2)
        self.assertTrue(np.allclose(arr1, arr2, atol=1e-05) and s1 == s2, msg)

    def _common_unary(self, shape, api, real, derive):
        data = np.random.rand(*shape) * 34
        var = eteq.variable(data, 'var')
        out = api(var)

        sess = eteq.Session()
        sess.track([out])
        sess.update()

        fout = out.get()
        self._array_close(real(data), fout)

        var2 = eteq.variable(data, 'var2')
        ex = eteq.derive(out, var)
        zero = eteq.derive(out, var2)

        sess.track([ex, zero])
        sess.update()

        data0 = np.zeros(shape, dtype=np.float32)
        der = ex.get()
        rej = zero.get()
        exdata = derive(data)
        self._array_close(exdata, der)
        self._array_eq(data0, rej)

    def _common_unary_tf(self, shape, api, tf_op):
        data = np.random.rand(*shape)
        var = eteq.variable(data, 'var')
        out = api(var)

        tf_var = tf.Variable(data)
        tf_out = tf_op(tf_var)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)

        sess = eteq.Session()
        sess.track([out])
        sess.update()

        fout = out.get()
        real = tfsess.run(tf_out)
        self._array_close(real, fout)

        var2 = eteq.variable(data, 'var2')
        ex = eteq.derive(out, var)
        zero = eteq.derive(out, var2)

        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex, zero])
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
        var = eteq.variable(data, 'var')
        var2 = eteq.variable(data2, 'var2')
        cst = random.uniform(0.5, 5)
        cst2 = random.uniform(0.5, 5)

        out = api(var, var2)
        both = api(var, var)
        clhs = api(var, cst)
        crhs = api(cst2, var2)

        sess = eteq.Session()
        sess.track([out, both, clhs, crhs])
        sess.update()

        fout = out.get()
        fboth = both.get()
        fclhs = clhs.get()
        fcrhs = crhs.get()
        self._array_close(real(data, data2), fout)
        self._array_close(real(data, data), fboth)
        self._array_close(real(data, cst), fclhs)
        self._array_close(real(cst2, data2), fcrhs)

        var3 = eteq.variable(data, 'var3')

        zero = eteq.derive(out, var3)
        ex = eteq.derive(out, var)
        ex2 = eteq.derive(out, var2)
        ex3 = eteq.derive(both, var)
        ex4 = eteq.derive(clhs, var)
        ex5 = eteq.derive(crhs, var2)

        sess.track([zero, ex, ex2, ex3, ex4, ex5])
        sess.update()

        rej = zero.get()
        der = ex.get()
        der2 = ex2.get()
        der3 = ex3.get()
        der4 = ex4.get()
        der5 = ex5.get()

        data0 = np.zeros(shape, dtype=np.float32)
        exdata = derive(0, (data, data2))
        exdata2 = derive(1, (data, data2))
        exdata3 = derive(0, (data, data)) + derive(1, (data, data))
        exdata4 = derive(0, (data, cst))
        exdata5 = derive(1, (cst2, data2))

        if isinstance(exdata4, float):
            exdata4 = np.array([exdata4] * np.prod(shape)).reshape(shape)
        if isinstance(exdata5, float):
            exdata5 = np.array([exdata5] * np.prod(shape)).reshape(shape)

        self._array_eq(data0, rej)
        self._array_close(exdata, der)
        self._array_close(exdata2, der2)
        self._array_close(exdata3, der3)
        self._array_close(exdata4, der4)
        self._array_close(exdata5, der5)

    def _common_reduce_1d(self, dim_reduce, tf_reduce):
        shape = [3, 4, 5]
        data = np.random.rand(*shape)
        var = eteq.variable(data, 'var')
        tf_var = tf.Variable(data)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)

        out = dim_reduce(var, 1)
        tf_out = tf_reduce(tf_var, [1])

        sess = eteq.Session()
        sess.track([out])
        sess.update()

        fout = out.get()
        tf_fout = tfsess.run(tf_out)

        self._array_close(tf_fout, fout)

        var2 = eteq.variable(data, 'var2')
        ex = eteq.derive(out, var)
        zero = eteq.derive(out, var2)
        sess.track([ex, zero])
        sess.update()

        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        data0 = np.zeros(shape, dtype=np.float32)
        der = ex.get()
        rej = zero.get()

        exdata = tfsess.run(tf_grad)

        self._array_close(exdata, der)
        self._array_eq(data0, rej)

    def _common_reduce(self, all_reduce, dim_reduce, tf_reduce):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            data = np.random.rand(*shape)
            var = eteq.variable(data, 'var')
            tf_var = tf.Variable(data)

            tfsess = tf.compat.v1.Session()
            tfsess.run(tf_var.initializer)

            out = all_reduce(var)
            out2 = dim_reduce(var, offset=1)
            tf_out = tf_reduce(tf_var)
            tf_out2 = tf_reduce(tf_var, [0, 1])

            sess = eteq.Session()
            sess.track([out, out2])
            sess.update()

            fout = out.get()
            fout2 = out2.get()
            tf_fout = np.array(tfsess.run(tf_out))
            tf_fout2 = tfsess.run(tf_out2)

            self._array_close(tf_fout, fout)
            self._array_close(tf_fout2, fout2)

            var2 = eteq.variable(data, 'var2')
            ex = eteq.derive(out, var)
            ex2 = eteq.derive(out2, var)
            zero = eteq.derive(out, var2)
            sess.track([ex, ex2, zero])
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
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            data0 = np.zeros(shape, dtype=np.float32)
            data = np.random.rand(*shape) * 234
            var = eteq.variable(data, 'var')

            sess = eteq.Session()
            sess.track([var])
            sess.update()
            fout = var.get()

            pad_removed = len(shape) - len(fout.shape)
            padding = [1] * pad_removed
            self.assertEqual(shape, padding + list(fout.shape))
            self._array_close(data, fout)

            var2 = eteq.variable(data, 'var2')
            one = eteq.derive(var, var)
            zero = eteq.derive(var, var2)
            sess.track([one, zero])
            sess.update()

            out1 = one.get()
            out0 = zero.get()
            self.assertEqual(shape, padding + list(out1.shape))
            self.assertEqual(shape, padding + list(out0.shape))
            self._array_eq(data1, out1)
            self._array_eq(data0, out0)

    def test_abs(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.abs, abs,
                lambda data: data / abs(data))

    def test_neg(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_unary(shape, tc.neg, lambda a: -a,
                lambda data: -data1)

    def test_sin(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.sin, np.sin, np.cos)

    def test_cos(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.cos, np.cos, lambda x: -np.sin(x))

    def test_tan(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.tan, tf.tan)

    def test_exp(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.exp, np.exp, np.exp)

    def test_log(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.log, np.log, lambda x: 1.0 / x)

    def test_sqrt(self):
        shape = [3, 4, 5]
        self._common_unary(shape, tc.sqrt, np.sqrt,
            lambda x: 1.0 / (2.0 * np.sqrt(x)))

    def test_round(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_unary(shape, tc.round, np.round, lambda x: data1)

    def test_sigmoid(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.sigmoid, tf.sigmoid)

    def test_tanh(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.tanh, tf.tanh)

    def test_clip_by_range(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape,
                lambda x: tc.clip_by_range(x, 0.3, 0.6),
                lambda x: tf.clip_by_value(x, 0.3, 0.6))

    def test_clip_by_l2norm(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape,
                lambda x: tc.clip_by_l2norm(x, 5),
                lambda x: tf.clip_by_norm(x, 5))

    def test_softmax(self):
        shapes = [[3, 4, 5]]
        if 'notvector.shape' in _test_data:
            shapes += _test_data['notvector.shape']
        for shape in shapes:
            self._common_unary_tf(shape, lambda arr: tc.softmax(arr,
                offset=0, ndims=1), tf.nn.softmax)
            self._common_unary_tf(shape, lambda arr: tc.softmax(arr,
                offset=1, ndims=1), lambda arr: tf.nn.softmax(arr, axis=len(shape)-2))

    def test_square(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.square, tf.square)

    def test_cube(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.cube, lambda x: tf.pow(x, 3))

    def test_pow(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        def pow_der(i, data):
            a, b = data
            if i == 0:
                return b * a ** (b - 1)
            return a ** b * np.log(a)
        for shape in shapes:
            self._common_binary(shape, tc.pow, lambda x, y: x ** y, pow_der)

    def test_add(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        generic_add = lambda x, y: x + y
        add_der = lambda i, data: data1
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_binary(shape, tc.add, generic_add, add_der)
            self._common_binary(shape, generic_add, generic_add, add_der)

    def test_sub(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        generic_sub = lambda x, y: x - y
        def sub_der(i, data):
            if i == 0:
                return data1
            return -data1
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_binary(shape, tc.sub, generic_sub, sub_der)
            self._common_binary(shape, generic_sub, generic_sub, sub_der)

    def test_mul(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        generic_mul = lambda x, y: x * y
        def mul_der(i, data):
            if i == 0:
                return data[1]
            return data[0]
        for shape in shapes:
            self._common_binary(shape, tc.mul, generic_mul, mul_der)
            self._common_binary(shape, generic_mul, generic_mul, mul_der)

    def test_div(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        generic_div = lambda x, y: x / y
        def div_der(i, data):
            a, b = data
            if i == 0:
                return 1 / b
            return -a / (b * b)
        for shape in shapes:
            self._common_binary(shape, tc.div, generic_div, div_der)
            self._common_binary(shape, generic_div, generic_div, div_der)

    def test_min(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        def min_der(i, data):
            a, b = data
            if i == 0:
                return (a <= b).astype(float)
            return (b <= a).astype(float)
        for shape in shapes:
            self._common_binary(shape, tc.min, np.minimum, min_der)

    def test_max(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        def max_der(i, data):
            a, b = data
            if i == 0:
                return (a >= b).astype(float)
            return (b >= a).astype(float)
        for shape in shapes:
            self._common_binary(shape, tc.max, np.maximum, max_der)

    def test_eq(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        np_eq = lambda x, y: np.round(x) == np.round(y)
        eq_der = lambda i, data: data0
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary(shape,
                lambda x, y: tc.eq(_round_helper(x), _round_helper(y)),
                np_eq, eq_der)
            self._common_binary(shape,
                lambda x, y: _round_helper(x) == _round_helper(y),
                np_eq, eq_der)

    def test_neq(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        np_neq = lambda x, y: np.round(x) != np.round(y)
        neq_der = lambda i, data: data0
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary(shape,
                lambda x, y: tc.neq(_round_helper(x), _round_helper(y)),
                np_neq, neq_der)
            self._common_binary(shape,
                lambda x, y: _round_helper(x) != _round_helper(y),
                np_neq, neq_der)

    def test_lt(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        np_lt = lambda x, y: np.round(x) < np.round(y)
        lt_der = lambda i, data: data0
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary(shape,
                lambda x, y: tc.lt(_round_helper(x), _round_helper(y)),
                np_lt, lt_der)
            self._common_binary(shape,
                lambda x, y: _round_helper(x) < _round_helper(y),
                np_lt, lt_der)

    def test_gt(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        np_gt = lambda x, y: np.round(x) > np.round(y)
        gt_der = lambda i, data: data0
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary(shape,
                lambda x, y: tc.gt(_round_helper(x), _round_helper(y)),
                np_gt, gt_der)
            self._common_binary(shape,
                lambda x, y: _round_helper(x) > _round_helper(y),
                np_gt, gt_der)

    def test_nelems(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_unary(shape, tc.n_elems,
                lambda data: np.prod(data.shape),
                lambda data: data0)

    def test_ndims(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_unary(shape,
                lambda x: tc.n_dims(x, 0),
                lambda data: data.shape[len(shape) - 1],
                lambda data: data0)

    def test_extend(self):
        shape = [2]
        data = np.random.rand(*shape) * 13
        expected_out = np.array(list(data) * 3).reshape([3, 2])
        var = eteq.variable(data, 'var')

        out = tc.extend(var, 1, [3])
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        fout = out.get()
        self._array_close(expected_out, fout)

        ex = eteq.derive(out, var)
        sess.track([ex])
        sess.update()

        der = ex.get()
        self._array_close(np.array([3, 3]), der)

    def test_rsum_1d(self):
        self._common_reduce_1d(tc.reduce_sum_1d, tf.reduce_sum)

    def test_rprod_1d(self):
        self._common_reduce_1d(tc.reduce_prod_1d, tf.reduce_prod)

    def test_rmin_1d(self):
        self._common_reduce_1d(tc.reduce_min_1d, tf.reduce_min)

    def test_rmax_1d(self):
        self._common_reduce_1d(tc.reduce_max_1d, tf.reduce_max)

    def test_rsum(self):
        self._common_reduce(tc.reduce_sum, tc.reduce_sum, tf.reduce_sum)

    def test_rprod(self):
        self._common_reduce(tc.reduce_prod, tc.reduce_prod, tf.reduce_prod)

    def test_rmin(self):
        self._common_reduce(tc.reduce_min, tc.reduce_min, tf.reduce_min)

    def test_rmax(self):
        self._common_reduce(tc.reduce_max, tc.reduce_max, tf.reduce_max)

    def test_rl2norm(self):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            data = np.random.rand(*shape)
            var = eteq.variable(data, 'var')
            tf_var = tf.Variable(data)

            tfsess = tf.compat.v1.Session()
            tfsess.run(tf_var.initializer)

            out = tc.reduce_l2norm(var)
            tf_out = tf.norm(tf_var)

            sess = eteq.Session()
            sess.track([out])
            sess.update()

            fout = out.get()
            tf_fout = np.array(tfsess.run(tf_out))

            self._array_close(tf_fout, fout)

            var2 = eteq.variable(data, 'var2')
            ex = eteq.derive(out, var)
            zero = eteq.derive(out, var2)
            sess.track([ex, zero])
            sess.update()

            tf_grad = tf.gradients(tf_out, [tf_var])[0]

            data0 = np.zeros(shape, dtype=np.float32)
            der = ex.get()
            rej = zero.get()

            exdata = tfsess.run(tf_grad)

            self._array_close(exdata, der)
            self._array_eq(data0, rej)

    def test_matmul(self):
        shapes = [
            ([5, 5], [5, 5]),
            ([50, 16], [16, 1])
        ]

        if 'matmul.shape_shape_left' in _test_data and\
            'matmul.shape_shape_right' in _test_data:
            lshape = _test_data['matmul.shape_shape_left'][0]
            rshape = _test_data['matmul.shape_shape_right'][0]
            rshape = [lshape[1]] + rshape
            shapes += [(lshape, rshape)]

        for lshape, rshape in shapes:
            is_symmetric = lshape == rshape
            if is_symmetric:
                outshape = lshape
            else:
                outshape = [rshape[0], lshape[1]]

            shape = [5, 5]
            data = np.random.rand(*lshape)
            data2 = np.random.rand(*rshape)

            var = eteq.variable(data, 'var')
            var2 = eteq.variable(data2, 'var2')
            tf_var = tf.Variable(data)
            tf_var2 = tf.Variable(data2)

            tfsess = tf.compat.v1.Session()
            tfsess.run(tf_var.initializer)
            tfsess.run(tf_var2.initializer)

            # regular matmul
            out = tc.matmul(var, var2)

            # tensorflow matmul
            tf_out = tf.matmul(tf_var, tf_var2)

            # evaluate regular matmul
            sess = eteq.Session()
            sess.track([out])
            sess.update()
            fout = out.get()

            # evaluate tensorflow matmul
            tf_fout = tfsess.run(tf_out)

            # check regular matmul
            self._array_close(tf_fout, fout)

            var3 = eteq.variable(data, 'var3')

            zero = eteq.derive(out, var3)
            ex = eteq.derive(out, var)
            ex2 = eteq.derive(out, var2)

            sess.track([zero, ex, ex2])
            sess.update()

            # eval derivative of regular matmul
            rej = zero.get()
            der = ex.get()
            der2 = ex2.get()

            # eval derivative of tensorflow matmul
            data0 = np.zeros(lshape, dtype=np.float32)
            tf_grad, tf_grad2 = tf.gradients(tf_out, [tf_var, tf_var2])

            exdata = tfsess.run(tf_grad)
            exdata2 = tfsess.run(tf_grad2)

            # check regular matmul
            self._array_eq(data0, rej)
            self._array_close(exdata, der)
            self._array_close(exdata2, der2)

            if is_symmetric:
                # test with symmetric property
                both = tc.matmul(var, var)

                tf_both = tf.matmul(tf_var, tf_var)

                sess.track([both])
                sess.update()
                fboth = both.get()

                tf_fboth = tfsess.run(tf_both)

                self._array_close(tf_fboth, fboth)

                ex3 = eteq.derive(both, var)
                sess.track([ex3])
                sess.update()

                der3 = ex3.get()

                tf_grad3 = tf.gradients(tf_both, [tf_var])[0]

                exdata3 = tfsess.run(tf_grad3)

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

            var = eteq.variable(data, 'var')
            vkernel = eteq.variable(kernel, 'vkernel')

            tf_var = tf.Variable(tf_data)
            tf_kernel = tf.Variable(tf_kdata)

            out = tc.convolution(var, vkernel, list(range(8)))

            sess = eteq.Session()
            sess.track([out])
            sess.update()

            fout = out.get()

            tfsess = tf.compat.v1.Session()
            tfsess.run(tf_var.initializer)
            tfsess.run(tf_kernel.initializer)

            tf_out = tf.nn.convolution(tf_var, tf_kernel, padding)
            tf_fout = tfsess.run(tf_out)

            tf_fout = tf_fout.reshape([tf_fout.shape[1], tf_fout.shape[2]])
            self._array_close(tf_fout, fout)

            var2 = eteq.variable(data, 'var2')
            zero = eteq.derive(out, var2)
            ex = eteq.derive(out, var)
            ex2 = eteq.derive(out, vkernel)

            sess.track([zero, ex, ex2])
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

    def test_conv2d(self):
        # batch, height, width, in
        shape = [3, 3, 3, 2]
        data = np.random.rand(*shape).astype(np.float32)

        # height, width, in, out
        kshape = [2, 2, 2, 4]
        kdata = np.random.rand(*kshape).astype(np.float32)

        image = eteq.variable(data, 'image')
        kernel = eteq.variable(kdata, 'vkernel')

        tfimage = tf.Variable(data)
        tfkernel = tf.Variable(kdata)

        out = tc.nn.conv2d(image, kernel)

        sess = eteq.Session()
        sess.track([out])
        sess.update()

        tfsess = tf.compat.v1.Session()

        tfoutput = tf.nn.conv2d(tfimage, tfkernel, [1, 1, 1, 1], 'VALID')
        tfsess.run(tfimage.initializer)
        tfsess.run(tfkernel.initializer)

        conv_output = out.get()
        tfconv_output = tfsess.run(tfoutput)
        self._array_close(tfconv_output, conv_output)

        var2 = eteq.variable(data, 'var2')
        zero = eteq.derive(out, var2)
        ex = eteq.derive(out, image)
        ex2 = eteq.derive(out, kernel)

        sess.track([zero, ex, ex2])
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

    def test_avgpool(self):
        # batch, height, width, in
        shape = [3, 8, 8, 2]
        data = np.random.rand(*shape).astype(np.float32)

        image = eteq.variable(data, 'image')
        out = tc.nn.mean_pool2d(image, [1, 2])
        sess = eteq.Session()
        sess.track([out])
        sess.update()
        output = out.get()

        tfsess = tf.compat.v1.Session()
        tfimage = tf.Variable(data)
        tfout = tf.nn.avg_pool2d(tfimage, [2, 2], [2, 2], padding='VALID')
        tfsess.run(tfimage.initializer)
        tf_output = tfsess.run(tfout)

        self._array_close(tf_output, output)

        # var2 = eteq.variable(data, 'var2')
        # zero = eteq.derive(out, var2)
        # ex = eteq.derive(out, image)

        # sess.track([zero, ex])
        # sess.update()

        # rej = zero.get()
        # der = ex.get()

        # data0 = np.zeros(shape, dtype=np.float32)
        # tf_grad, tf_grad2 = tf.gradients(tfoutput, [tfimage, tfkernel])

        # exdata = tfsess.run(tf_grad)
        # exdata2 = tfsess.run(tf_grad2)

        # self._array_eq(data0, rej)
        # self._array_close(exdata, der)

    def test_maxpool(self):
        shape = [3, 8, 8, 2]
        data = np.random.rand(*shape).astype(np.float32)

        image = eteq.variable(data, 'image')
        out = tc.nn.max_pool2d(image, [1, 2])
        sess = eteq.Session()
        sess.track([out])
        sess.update()
        output = out.get()

        tfsess = tf.compat.v1.Session()
        tfimage = tf.Variable(data)
        tfout = tf.nn.max_pool2d(tfimage, [2, 2], [2, 2], padding='VALID')
        tfsess.run(tfimage.initializer)
        tf_output = tfsess.run(tfout)

        self._array_close(tf_output, output)

        # var2 = eteq.variable(data, 'var2')
        # zero = eteq.derive(out, var2)
        # ex = eteq.derive(out, image)

        # sess.track([zero, ex])
        # sess.update()

        # rej = zero.get()
        # der = ex.get()

        # data0 = np.zeros(shape, dtype=np.float32)
        # tf_grad, tf_grad2 = tf.gradients(tfoutput, [tfimage, tfkernel])

        # exdata = tfsess.run(tf_grad)
        # exdata2 = tfsess.run(tf_grad2)

        # self._array_eq(data0, rej)
        # self._array_close(exdata, der)

    def test_grader_scenario1(self): # REDUCE -> MUL
        data = np.random.rand(3,10)
        data2 = np.random.rand(10)

        var = eteq.variable(data, 'var')
        var2 = eteq.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.mul(tc.reduce_sum(var, offset=1, ndims=1), var2)
        tf_out = tf.multiply(tf.reduce_sum(tf_var, 0), tf_var2)

        # evaluate regular matmul
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = eteq.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex])
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario2(self): # EXTEND -> MUL
        data = np.random.rand(10)
        data2 = np.random.rand(3,10)

        var = eteq.variable(data, 'var')
        var2 = eteq.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.mul(tc.extend(var, 1, [3]), var2)
        tf_out = tf_var * tf_var2

        # evaluate regular matmul
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = eteq.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex])
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario3(self): # PERMUTE -> MUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,10)

        var = eteq.variable(data, 'var')
        var2 = eteq.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.mul(tc.permute(var, [1,0]), var2)
        tf_out = tf.transpose(tf_var) * tf_var2

        # evaluate regular matmul
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = eteq.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex])
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario4(self): # MATMUL -> MUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,5)
        data3 = np.random.rand(10,5)

        var = eteq.variable(data, 'var')
        var2 = eteq.variable(data2, 'var2')
        var3 = eteq.variable(data3, 'var3')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_var3 = tf.Variable(data3)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)
        tfsess.run(tf_var3.initializer)

        out = tc.mul(tc.matmul(var, var2), var3)
        tf_out = tf.multiply(tf.matmul(tf_var, tf_var2), tf_var3)

        # evaluate regular matmul
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = eteq.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex])
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario5(self): # MATMUL -> MATMUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,5)
        data3 = np.random.rand(5,4)

        var = eteq.variable(data, 'var')
        var2 = eteq.variable(data2, 'var2')
        var3 = eteq.variable(data3, 'var3')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_var3 = tf.Variable(data3)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)
        tfsess.run(tf_var3.initializer)

        out = tc.matmul(tc.matmul(var, var2), var3)
        tf_out = tf.matmul(tf.matmul(tf_var, tf_var2), tf_var3)

        # evaluate regular matmul
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = eteq.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex])
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario6(self): # REDUCE -> MATMUL
        data = np.random.rand(4,10,3)
        data2 = np.random.rand(3,5)

        var = eteq.variable(data, 'var')
        var2 = eteq.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.matmul(tc.reduce_sum(var, offset=2, ndims=1), var2)
        tf_out = tf.matmul(tf.reduce_sum(tf_var, 0), tf_var2)

        # evaluate regular matmul
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = eteq.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex])
        sess.update()

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario7(self): # EXTEND -> MATMUL
        data = np.random.rand(3)
        data2 = np.random.rand(3,5)
        ones = np.ones([10, 3])

        var = eteq.variable(data, 'var')
        var2 = eteq.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.matmul(tc.extend(var, 1, [10]), var2)
        tf_out = tf.matmul(tf_var * ones, tf_var2)

        # evaluate regular matmul
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = eteq.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex])
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

        var = eteq.variable(data, 'var')
        tf_var = tf.Variable(data)

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_var.initializer)

        out = tc.matmul(
            tc.extend(var, 1, [10]),
            tc.permute(tc.extend(var, 1, [3]), [1, 0]))
        tf_out = tf.matmul(tf_var * ones, tf.transpose(tf_var * ones2))

        # evaluate regular matmul
        sess = eteq.Session()
        sess.track([out])
        sess.update()

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = eteq.derive(out, var)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess.track([ex])
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

        a = eteq.variable(data, 'a')
        b = eteq.variable(data2, 'b')
        c = eteq.variable(data3, 'c')
        tf_a = tf.Variable(data)
        tf_b = tf.Variable(data2)
        tf_c = tf.Variable(data3)

        d = tc.matmul(a, b)
        e = tc.matmul(c, d)
        f = tc.matmul(tc.transpose(d), tc.transpose(c))
        dest = tc.matmul(e, f)

        tf_d = tf.matmul(tf_a, tf_b)
        tf_e = tf.matmul(tf_c, tf_d)
        tf_f = tf.matmul(tf.transpose(tf_d), tf.transpose(tf_c))
        tf_dest = tf.matmul(tf_e, tf_f)

        da = eteq.derive(dest, a)
        db = eteq.derive(dest, b)
        dc = eteq.derive(dest, c)
        tf_da, tf_db, tf_dc = tf.gradients(tf_dest, [tf_a, tf_b, tf_c])

        tfsess = tf.compat.v1.Session()
        tfsess.run(tf_a.initializer)
        tfsess.run(tf_b.initializer)
        tfsess.run(tf_c.initializer)

        sess = eteq.Session()
        sess.track([dest, da, db, dc])
        sess.update()

        exa = tfsess.run(tf_da)
        exb = tfsess.run(tf_db)
        exc = tfsess.run(tf_dc)
        self._array_close(exa, da.get())
        self._array_close(exb, db.get())
        self._array_close(exc, dc.get())

if __name__ == "__main__":
    with open('testutil/ead_template.json') as json_data:
        test_template = json.load(json_data)
        assert 'test_cases' in test_template
        assert 'config_pools' in test_template

    # log to file
    logging.basicConfig(filename='/tmp/ead_ptest.log',level=logging.DEBUG)
    logging.info("running ptest for eteq")

    _test_data = generate_testcases(
        test_template['test_cases'],
        test_template['config_pools'])

    unittest.main()
