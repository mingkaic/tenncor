import json
import random
import logging
import unittest
import numpy as np
import tensorflow as tf

import tenncor as tc

from testutil.array_testcase import ArrTest
from testutil.generate_testcases import generate_testcases
from testutil.tf_testutil import tf_init, version_lt

tfSess = tf_init()

_test_data = {}

def _round_helper(x):
    if isinstance(x, float):
        return round(x)
    return tc.api.round(x)

class EADTest(ArrTest):
    def _common_assign(self, shape, api, real):
        data = np.random.rand(*shape) * 34
        data2 = np.random.rand(*shape) * 13
        var = tc.variable(data, 'var_target')
        var2 = tc.variable(data, 'var_target2')
        src = tc.variable(data2, 'var_source')

        ass1 = api(var, src)
        ass2 = api(var2, tc.api.neg(src))

        expect_a1 = real(data, data2)
        expect_a2 = real(data, -data2)

        self._array_close(expect_a1, ass1.get())
        self._array_close(expect_a2, ass2.get())

    def _common_unary(self, shape, api, real, derive):
        data = np.random.rand(*shape) * 34
        var = tc.variable(data, 'var')
        out = api(var)

        fout = out.get()
        self._array_close(real(data), fout)

        var2 = tc.variable(data, 'var2')
        ex, zero = tuple(tc.derive(out, [var, var2]))

        data0 = np.zeros(shape, dtype=np.float32)
        der = ex.get()
        rej = zero.get()
        exdata = derive(data)
        self._array_close(exdata, der)
        self._array_eq(data0, rej)

    def _common_unary_tf(self, shape, api, tf_op):
        data = np.random.rand(*shape)
        var = tc.variable(data, 'var')
        out = api(var)

        tf_var = tf.Variable(data)
        tf_out = tf_op(tf_var)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)

        fout = out.get()
        real = tfsess.run(tf_out)
        self._array_close(real, fout)

        var2 = tc.variable(data, 'var2')
        ex, zero = tuple(tc.derive(out, [var, var2]))

        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        data0 = np.zeros(shape, dtype=np.float32)
        der = ex.get()
        rej = zero.get()
        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, der)
        self._array_eq(data0, rej)

    def _common_binary(self, shape, api, real, derive):
        data = np.random.rand(*shape)
        data2 = np.random.rand(*shape)
        var = tc.variable(data, 'var')
        var2 = tc.variable(data2, 'var2')
        cst = random.uniform(0.5, 5)
        cst2 = random.uniform(0.5, 5)

        out = api(var, var2)
        both = api(var, var)
        clhs = api(var, cst)
        crhs = api(cst2, var2)

        fout = out.get()
        fboth = both.get()
        fclhs = clhs.get()
        fcrhs = crhs.get()
        self._array_close(real(data, data2), fout)
        self._array_close(real(data, data), fboth)
        self._array_close(real(data, cst), fclhs)
        self._array_close(real(cst2, data2), fcrhs)

        var3 = tc.variable(data, 'var3')

        zero, ex, ex2 = tuple(tc.derive(out, [var3, var, var2]))
        ex3 = tc.derive(both, [var])[0]
        ex4 = tc.derive(clhs, [var])[0]
        ex5 = tc.derive(crhs, [var2])[0]

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
        var = tc.variable(data, 'var')
        tf_var = tf.Variable(data)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)

        out = dim_reduce(var, 1)
        tf_out = tf_reduce(tf_var, [1])

        fout = out.get()
        tf_fout = tfsess.run(tf_out)

        self._array_close(tf_fout, fout)

        var2 = tc.variable(data, 'var2')
        ex, zero = tuple(tc.derive(out, [var, var2]))

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
            var = tc.variable(data, 'var')
            tf_var = tf.Variable(data)

            tfsess = tfSess()
            tfsess.run(tf_var.initializer)

            out = all_reduce(var)
            out2 = dim_reduce(var, offset=1)
            tf_out = tf_reduce(tf_var)
            tf_out2 = tf_reduce(tf_var, [0, 1])

            fout = out.get()
            fout2 = out2.get()
            tf_fout = np.array(tfsess.run(tf_out))
            tf_fout2 = tfsess.run(tf_out2)

            self._array_close(tf_fout, fout)
            self._array_close(tf_fout2, fout2)

            var2 = tc.variable(data, 'var2')
            ex, zero = tuple(tc.derive(out, [var, var2]))
            ex2 = tc.derive(out2, [var])[0]

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

    def _common_argreduce(self, dim_reduce, tf_reduce):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            data = np.random.rand(*shape)
            var = tc.variable(data, 'var')
            tf_var = tf.Variable(data)

            tfsess = tfSess()
            tfsess.run(tf_var.initializer)

            out = tc.api.permute(dim_reduce(var,
                return_dim=1), [0, 2, 1])
            tf_out = tf_reduce(tf_var, 1)

            fout = out.get()
            tf_fout = tfsess.run(tf_out)

            self._array_close(tf_fout, fout)
            # arg reduce has no derivatives

    def test_variable(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            data0 = np.zeros(shape, dtype=np.float32)
            data = np.random.rand(*shape) * 234
            var = tc.variable(data, 'var')

            fout = var.get()

            pad_removed = len(shape) - len(fout.shape)
            padding = [1] * pad_removed
            self.assertEqual(shape, padding + list(fout.shape))
            self._array_close(data, fout)

            var2 = tc.variable(data, 'var2')
            one, zero = tuple(tc.derive(var, [var, var2]))

            out1 = one.get()
            out0 = zero.get()
            self.assertEqual(shape, padding + list(out1.shape))
            self.assertEqual(shape, padding + list(out0.shape))
            self._array_eq(data1, out1)
            self._array_eq(data0, out0)

    def test_assign(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_assign(shape, tc.api.assign, lambda t, s: s)

    def test_assign_add(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_assign(shape, tc.api.assign_add, lambda t, s: t + s)

    def test_assign_sub(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_assign(shape, tc.api.assign_sub, lambda t, s: t - s)

    def test_assign_mul(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_assign(shape, tc.api.assign_mul, lambda t, s: t * s)

    def test_assign_div(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_assign(shape, tc.api.assign_div, lambda t, s: t / s)

    def test_abs(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.api.abs, abs,
                lambda data: data / abs(data))

    def test_neg(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_unary(shape, tc.api.neg, lambda a: -a,
                lambda data: -data1)

    def test_sin(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.api.sin, np.sin, np.cos)

    def test_cos(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.api.cos, np.cos, lambda x: -np.sin(x))

    def test_tan(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.api.tan, tf.tan)

    def test_exp(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.api.exp, np.exp, np.exp)

    def test_log(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary(shape, tc.api.log, np.log, lambda x: 1.0 / x)

    def test_sqrt(self):
        shape = [3, 4, 5]
        self._common_unary(shape, tc.api.sqrt, np.sqrt,
            lambda x: 1.0 / (2.0 * np.sqrt(x)))

    def test_round(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_unary(shape, tc.api.round, np.round, lambda x: data1)

    def test_sigmoid(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.api.sigmoid, tf.sigmoid)

    def test_tanh(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.api.tanh, tf.tanh)

    def test_clip_by_range(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape,
                lambda x: tc.api.clip_by_range(x, 0.3, 0.6),
                lambda x: tf.clip_by_value(x, 0.3, 0.6))

    def test_clip_by_l2norm(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape,
                lambda x: tc.api.clip_by_l2norm(x, 5),
                lambda x: tf.clip_by_norm(x, 5))

    def test_softmax(self):
        shapes = [[3, 4, 5]]
        if 'notvector.shape' in _test_data:
            shapes += _test_data['notvector.shape']
        for shape in shapes:
            self._common_unary_tf(shape, lambda arr: tc.api.softmax(arr,
                offset=0, ndims=1), tf.nn.softmax)
            self._common_unary_tf(shape, lambda arr: tc.api.softmax(arr,
                offset=1, ndims=1), lambda arr: tf.nn.softmax(arr, axis=len(shape)-2))

    def test_relu(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.api.relu, tf.nn.relu)

    def test_square(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.api.square, tf.square)

    def test_cube(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            self._common_unary_tf(shape, tc.api.cube, lambda x: tf.pow(x, 3))

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
            self._common_binary(shape, tc.api.pow, lambda x, y: x ** y, pow_der)

    def test_add(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        generic_add = lambda x, y: x + y
        add_der = lambda i, data: data1
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_binary(shape, tc.api.add, generic_add, add_der)
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
            self._common_binary(shape, tc.api.sub, generic_sub, sub_der)
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
            self._common_binary(shape, tc.api.mul, generic_mul, mul_der)
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
            self._common_binary(shape, tc.api.div, generic_div, div_der)
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
            self._common_binary(shape, tc.api.min, np.minimum, min_der)

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
            self._common_binary(shape, tc.api.max, np.maximum, max_der)

    def test_eq(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        np_eq = lambda x, y: np.round(x) == np.round(y)
        eq_der = lambda i, data: data0
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary(shape,
                lambda x, y: tc.api.eq(_round_helper(x), _round_helper(y)),
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
                lambda x, y: tc.api.neq(_round_helper(x), _round_helper(y)),
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
                lambda x, y: tc.api.lt(_round_helper(x), _round_helper(y)),
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
                lambda x, y: tc.api.gt(_round_helper(x), _round_helper(y)),
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
            self._common_unary(shape, tc.api.n_elems,
                lambda data: np.prod(data.shape),
                lambda data: data0)

    def test_ndims(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _test_data:
            shapes += _test_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_unary(shape,
                lambda x: tc.api.n_dims(x, 0),
                lambda data: data.shape[len(shape) - 1],
                lambda data: data0)

    def test_extend(self):
        shape = [2]
        data = np.random.rand(*shape) * 13
        expected_out = np.array(list(data) * 3).reshape([3, 2])
        var = tc.variable(data, 'var')

        out = tc.api.extend(var, 1, [3])

        fout = out.get()
        self._array_close(expected_out, fout)

        ex = tc.derive(out, [var])[0]

        der = ex.get()
        self._array_close(np.array([3, 3]), der)

    def test_rsum_1d(self):
        self._common_reduce_1d(tc.api.reduce_sum_1d, tf.reduce_sum)

    def test_rprod_1d(self):
        self._common_reduce_1d(tc.api.reduce_prod_1d, tf.reduce_prod)

    def test_rmin_1d(self):
        self._common_reduce_1d(tc.api.reduce_min_1d, tf.reduce_min)

    def test_rmax_1d(self):
        self._common_reduce_1d(tc.api.reduce_max_1d, tf.reduce_max)

    def test_rsum(self):
        self._common_reduce(tc.api.reduce_sum, tc.api.reduce_sum, tf.reduce_sum)

    def test_rprod(self):
        self._common_reduce(tc.api.reduce_prod, tc.api.reduce_prod, tf.reduce_prod)

    def test_rmin(self):
        self._common_reduce(tc.api.reduce_min, tc.api.reduce_min, tf.reduce_min)

    def test_rmax(self):
        self._common_reduce(tc.api.reduce_max, tc.api.reduce_max, tf.reduce_max)

    def test_argmax(self):
        self._common_argreduce(tc.api.argmax, tf.argmax)

    def test_rl2norm(self):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            data = np.random.rand(*shape)
            var = tc.variable(data, 'var')
            tf_var = tf.Variable(data)

            tfsess = tfSess()
            tfsess.run(tf_var.initializer)

            out = tc.api.reduce_l2norm(var)
            tf_out = tf.norm(tf_var)

            fout = out.get()
            tf_fout = np.array(tfsess.run(tf_out))

            self._array_close(tf_fout, fout)

            var2 = tc.variable(data, 'var2')
            ex, zero = tuple(tc.derive(out, [var, var2]))

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

            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            tf_var = tf.Variable(data)
            tf_var2 = tf.Variable(data2)

            tfsess = tfSess()
            tfsess.run(tf_var.initializer)
            tfsess.run(tf_var2.initializer)

            # regular matmul
            out = tc.api.matmul(var, var2)

            # tensorflow matmul
            tf_out = tf.matmul(tf_var, tf_var2)

            # evaluate regular matmul
            fout = out.get()

            # evaluate tensorflow matmul
            tf_fout = tfsess.run(tf_out)

            # check regular matmul
            self._array_close(tf_fout, fout)

            var3 = tc.variable(data, 'var3')

            zero, ex, ex2 = tuple(tc.derive(out, [var3, var, var2]))

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
                both = tc.api.matmul(var, var)

                tf_both = tf.matmul(tf_var, tf_var)

                fboth = both.get()

                tf_fboth = tfsess.run(tf_both)

                self._array_close(tf_fboth, fboth)

                ex3 = tc.derive(both, [var])[0]

                der3 = ex3.get()

                tf_grad3 = tf.gradients(tf_both, [tf_var])[0]

                exdata3 = tfsess.run(tf_grad3)

                self._array_close(exdata3, der3)

    def test_convolution(self):
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

            var = tc.variable(data, 'var')
            vkernel = tc.variable(kernel, 'vkernel')

            tf_var = tf.Variable(tf_data)
            tf_kernel = tf.Variable(tf_kdata)

            out = tc.api.convolution(var, vkernel, list(range(8)))

            fout = out.get()

            tfsess = tfSess()
            tfsess.run(tf_var.initializer)
            tfsess.run(tf_kernel.initializer)

            tf_out = tf.nn.convolution(tf_var, tf_kernel, padding='VALID')
            tf_fout = tfsess.run(tf_out)

            tf_fout = tf_fout.reshape([tf_fout.shape[1], tf_fout.shape[2]])
            self._array_close(tf_fout, fout)

            var2 = tc.variable(data, 'var2')
            zero, ex, ex2 = tuple(tc.derive(out, [var2, var, vkernel]))

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

        image = tc.variable(data, 'image')
        kernel = tc.variable(kdata, 'vkernel')

        tfimage = tf.Variable(data)
        tfkernel = tf.Variable(kdata)

        out = tc.api.nn.conv2d(image, kernel)

        tfsess = tfSess()

        tfoutput = tf.nn.conv2d(tfimage, tfkernel, [1, 1, 1, 1], 'VALID')
        tfsess.run(tfimage.initializer)
        tfsess.run(tfkernel.initializer)

        conv_output = out.get()
        tfconv_output = tfsess.run(tfoutput)
        self._array_close(tfconv_output, conv_output)

        var2 = tc.variable(data, 'var2')
        zero, ex, ex2 = tuple(tc.derive(out, [var2, image, kernel]))

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

        image = tc.variable(data, 'image')
        out = tc.api.nn.mean_pool2d(image, [1, 2])
        output = out.get()

        tfsess = tfSess()
        tfimage = tf.Variable(data)
        tfout = tf.nn.avg_pool2d(tfimage, [2, 2], [2, 2], padding='VALID')
        tfsess.run(tfimage.initializer)
        tfoutput = tfsess.run(tfout)

        self._array_close(tfoutput, output)

        var2 = tc.variable(data, 'var2')
        zero, ex = tuple(tc.derive(out, [var2, image]))

        rej = zero.get()
        der = ex.get()

        data0 = np.zeros(shape, dtype=np.float32)
        tf_grad = tf.gradients(tfout, [tfimage])

        exdata = tfsess.run(tf_grad)

        self._array_eq(data0, rej)
        self._array_close(exdata[0], der)

    def test_stride(self):
        shape = [3, 8, 8, 2]
        data = np.random.rand(*shape).astype(np.float32)
        image = tc.variable(data, 'image')

        strideout = tc.api.stride(image, [1, 2, 2])
        self._array_eq([3, 4, 4, 2], strideout.shape())

        ex = tc.derive(strideout, [image])[0]
        self._array_eq(shape, ex.shape())

    def test_maxpool(self):
        shape = [3, 8, 8, 2]
        data = np.random.rand(*shape).astype(np.float32)

        image = tc.variable(data, 'image')
        out = tc.api.nn.max_pool2d(image, [1, 2])
        output = out.get()

        tfsess = tfSess()
        tfimage = tf.Variable(data)
        tfout = tf.nn.max_pool2d(tfimage, [2, 2], [2, 2], padding='VALID')
        tfsess.run(tfimage.initializer)
        tfoutput = tfsess.run(tfout)

        self._array_close(tfoutput, output)

        var2 = tc.variable(data, 'var2')
        zero, ex = tuple(tc.derive(out, [var2, image]))

        rej = zero.get()
        der = ex.get()

        data0 = np.zeros(shape, dtype=np.float32)
        tf_grad = tf.gradients(tfout, [tfimage])

        exdata = tfsess.run(tf_grad)

        self._array_eq(data0, rej)
        self._array_close(exdata[0], der)

    def test_grader_scenario1(self): # REDUCE -> MUL
        data = np.random.rand(3,10)
        data2 = np.random.rand(10)

        var = tc.variable(data, 'var')
        var2 = tc.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.api.mul(tc.api.reduce_sum(var, offset=1, ndims=1), var2)
        tf_out = tf.multiply(tf.reduce_sum(tf_var, 0), tf_var2)

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = tc.derive(out, [var])[0]
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario2(self): # EXTEND -> MUL
        data = np.random.rand(10)
        data2 = np.random.rand(3,10)

        var = tc.variable(data, 'var')
        var2 = tc.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.api.mul(tc.api.extend(var, 1, [3]), var2)
        tf_out = tf_var * tf_var2

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = tc.derive(out, [var])[0]
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario3(self): # PERMUTE -> MUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,10)

        var = tc.variable(data, 'var')
        var2 = tc.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.api.mul(tc.api.permute(var, [1,0]), var2)
        tf_out = tf.transpose(tf_var) * tf_var2

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = tc.derive(out, [var])[0]
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario4(self): # MATMUL -> MUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,5)
        data3 = np.random.rand(10,5)

        var = tc.variable(data, 'var')
        var2 = tc.variable(data2, 'var2')
        var3 = tc.variable(data3, 'var3')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_var3 = tf.Variable(data3)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)
        tfsess.run(tf_var3.initializer)

        out = tc.api.mul(tc.api.matmul(var, var2), var3)
        tf_out = tf.multiply(tf.matmul(tf_var, tf_var2), tf_var3)

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = tc.derive(out, [var])[0]
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario5(self): # MATMUL -> MATMUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,5)
        data3 = np.random.rand(5,4)

        var = tc.variable(data, 'var')
        var2 = tc.variable(data2, 'var2')
        var3 = tc.variable(data3, 'var3')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_var3 = tf.Variable(data3)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)
        tfsess.run(tf_var3.initializer)

        out = tc.api.matmul(tc.api.matmul(var, var2), var3)
        tf_out = tf.matmul(tf.matmul(tf_var, tf_var2), tf_var3)

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = tc.derive(out, [var])[0]
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario6(self): # REDUCE -> MATMUL
        data = np.random.rand(4,10,3)
        data2 = np.random.rand(3,5)

        var = tc.variable(data, 'var')
        var2 = tc.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.api.matmul(tc.api.reduce_sum(var, offset=2, ndims=1), var2)
        tf_out = tf.matmul(tf.reduce_sum(tf_var, 0), tf_var2)

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = tc.derive(out, [var])[0]
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario7(self): # EXTEND -> MATMUL
        data = np.random.rand(3)
        data2 = np.random.rand(3,5)
        ones = np.ones([10, 3])

        var = tc.variable(data, 'var')
        var2 = tc.variable(data2, 'var2')
        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)
        tfsess.run(tf_var2.initializer)

        out = tc.api.matmul(tc.api.extend(var, 1, [10]), var2)
        tf_out = tf.matmul(tf_var * ones, tf_var2)

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = tc.derive(out, [var])[0]
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    # A<a> -> EXTEND -> B<a,b>
    # A<a> -> EXTEND -> C<a,c> -> PERMUTE -> <c,a>
    # B MATMUL C -> D<c,b>
    def test_grader_scenario8(self):
        data = np.random.rand(5)
        ones = np.ones([10, 5])
        ones2 = np.ones([3, 5])

        var = tc.variable(data, 'var')
        tf_var = tf.Variable(data)

        tfsess = tfSess()
        tfsess.run(tf_var.initializer)

        out = tc.api.matmul(
            tc.api.extend(var, 1, [10]),
            tc.api.permute(tc.api.extend(var, 1, [3]), [1, 0]))
        tf_out = tf.matmul(tf_var * ones, tf.transpose(tf_var * ones2))

        # evaluate tensorflow matmul
        tf_fout = tfsess.run(tf_out)

        # check regular matmul
        self._array_close(tf_fout, out.get())

        ex = tc.derive(out, [var])[0]
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        exdata = tfsess.run(tf_grad)
        self._array_close(exdata, ex.get())

    def test_grader_scenario9(self):
        ashape = [2, 3]
        bshape = [3, 4]
        cshape = [4, 2]

        data = np.random.rand(*ashape)
        data2 = np.random.rand(*bshape)
        data3 = np.random.rand(*cshape)

        a = tc.variable(data, 'a')
        b = tc.variable(data2, 'b')
        c = tc.variable(data3, 'c')
        tf_a = tf.Variable(data)
        tf_b = tf.Variable(data2)
        tf_c = tf.Variable(data3)

        d = tc.api.matmul(a, b)
        e = tc.api.matmul(c, d)
        f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))
        dest = tc.api.matmul(e, f)

        tf_d = tf.matmul(tf_a, tf_b)
        tf_e = tf.matmul(tf_c, tf_d)
        tf_f = tf.matmul(tf.transpose(tf_d), tf.transpose(tf_c))
        tf_dest = tf.matmul(tf_e, tf_f)

        da, db, dc = tuple(tc.derive(dest, [a, b, c]))
        tf_da, tf_db, tf_dc = tf.gradients(tf_dest, [tf_a, tf_b, tf_c])

        tfsess = tfSess()
        tfsess.run(tf_a.initializer)
        tfsess.run(tf_b.initializer)
        tfsess.run(tf_c.initializer)

        exa = tfsess.run(tf_da)
        exb = tfsess.run(tf_db)
        exc = tfsess.run(tf_dc)
        self._array_close(exa, da.get())
        self._array_close(exb, db.get())
        self._array_close(exc, dc.get())

if version_lt(tf.__version__, '1.6.0'):
    delattr(EADTest, 'test_maxpool')
    delattr(EADTest, 'test_avgpool')

if __name__ == "__main__":
    with open('testutil/ead_template.json') as json_data:
        test_template = json.load(json_data)
        assert 'test_cases' in test_template
        assert 'config_pools' in test_template

    # log to file
    logging.basicConfig(filename='/tmp/ead_ptest.log',level=logging.DEBUG)
    logging.info("running ptest for tc")

    _test_data = generate_testcases(
        test_template['test_cases'],
        test_template['config_pools'])

    unittest.main()
