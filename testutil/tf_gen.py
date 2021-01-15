
import random
import json
import sys
import logging

import numpy as np
import tensorflow as tf
from testutil.tf_testutil import tf_init, version_lt
from testutil.generate_testcases import generate_testcases
from testutil.compare_testcase import TestWriter

Session = tf_init()

_generate_data = {}

class TFWriter(TestWriter):
    def __init__(self, outfile):
        super().__init__(outfile)

    def _common_assign(self, case, shape, api):
        data = np.random.rand(*shape) * 34
        data2 = np.random.rand(*shape) * 13

        a1 = api(data, data2)
        a2 = api(data, -data2)

        self.save_case(case + '_pos', (data, data2), a1)
        self.save_case(case + '_neg', (data, data2), a2)

    def _common_unary_raw(self, case, shape, api, derive):
        data = np.random.rand(*shape) * 34

        fwd = api(data)
        bwd = derive(data)

        self.save_case(case + '_fwd', (data), fwd)
        self.save_case(case + '_bwd', (data), bwd)

    def _common_unary(self, case, shape, api):
        data = np.random.rand(*shape)

        var = tf.Variable(data)
        out = api(var)
        grad = tf.gradients(out, [var])[0]

        sess = Session()
        sess.run(var.initializer)

        fwd = sess.run(out)
        bwd = sess.run(grad)

        self.save_case(case + '_fwd', (data), fwd)
        self.save_case(case + '_bwd', (data), bwd)

    def _common_unary_nograd(self, case, shape, tf_reduce):
        data = np.random.rand(*shape)
        var = tf.Variable(data)

        sess = Session()
        sess.run(var.initializer)

        out = tf_reduce(var, 1)

        fwd = sess.run(out)

        self.save_case(case, (data), fwd)

    def _common_binary(self, case, shape, api, derive):
        data = np.random.rand(*shape)
        data2 = np.random.rand(*shape)
        cst = random.uniform(0.5, 5)
        cst2 = random.uniform(0.5, 5)

        outargs = (data, data2)
        bothargs = (data, data)
        clhsargs = (data, cst)
        crhsargs = (cst2, data)

        out = api(*outargs)
        both = api(*bothargs)
        clhs = api(*clhsargs)
        crhs = api(*crhsargs)

        outda = derive(0, outargs)
        outdb = derive(1, outargs)
        bothd = derive(0, bothargs) + derive(1, bothargs)
        clhsd = derive(0, clhsargs)
        crhsd = derive(1, crhsargs)

        if isinstance(clhsd, float):
            clhsd = np.array([clhsd] * np.prod(shape)).reshape(shape)
        if isinstance(crhsd, float):
            crhsd = np.array([crhsd] * np.prod(shape)).reshape(shape)

        self.save_case(case + '_out', outargs, out)
        self.save_case(case + '_both', (data), both)
        self.save_case(case + '_clhs', clhsargs, clhs)
        self.save_case(case + '_crhs', crhsargs, crhs)

        self.save_case(case + '_outda', outargs, outda)
        self.save_case(case + '_outdb', outargs, outdb)
        self.save_case(case + '_bothd', (data), bothd)
        self.save_case(case + '_clhsd', clhsargs, clhsd)
        self.save_case(case + '_crhsd', crhsargs, crhsd)

    def run_all(self):
        for method in dir(self):
            if method.startswith('generate_'):
                getattr(self, method)()

    def generate_assign(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_assign('assign', shape, lambda t, s: s)

    def generate_assign_add(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_assign('assign_add', shape, lambda t, s: t + s)

    def generate_assign_sub(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_assign('assign_sub', shape, lambda t, s: t - s)

    def generate_assign_mul(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_assign('assign_mul', shape, lambda t, s: t * s)

    def generate_assign_div(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_assign('assign_div', shape, lambda t, s: t / s)

    def generate_abs(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary_raw('abs', shape, abs, lambda data: data / abs(data))

    def generate_neg(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_unary_raw('neg', shape, lambda a: -a, lambda data: -data1)

    def generate_sin(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary_raw('sin', shape, np.sin, np.cos)

    def generate_cos(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary_raw('cos', shape, np.cos, lambda x: -np.sin(x))

    def generate_tan(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary('tan', shape, tf.tan)

    def generate_exp(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary_raw('exp', shape, np.exp, np.exp)

    def generate_log(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary_raw('log', shape, np.log, lambda x: 1.0 / x)

    def generate_sqrt(self):
        shape = [3, 4, 5]
        self._common_unary_raw('sqrt', shape, np.sqrt,
            lambda x: 1.0 / (2.0 * np.sqrt(x)))

    def generate_round(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_unary_raw('round', shape, np.round, lambda x: data1)

    def generate_sigmoid(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary('sigmoid', shape, tf.sigmoid)

    def generate_tanh(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary('tanh', shape, tf.tanh)

    def generate_clip_by_range(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary('clip_by_range', shape,
                lambda x: tf.clip_by_value(x, 0.3, 0.6))

    def generate_clip_by_l2norm(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary('clip_by_l2norm', shape,
                lambda x: tf.clip_by_norm(x, 5))

    def generate_softmax(self):
        shapes = [[3, 4, 5]]
        if 'notvector.shape' in _generate_data:
            shapes += _generate_data['notvector.shape']
        for shape in shapes:
            self._common_unary('softmax0', shape, tf.nn.softmax)
            self._common_unary('softmax1', shape, lambda arr: tf.nn.softmax(arr, axis=len(shape)-2))

    def generate_relu(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary('relu', shape, tf.nn.relu)

    def generate_square(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary('square', shape, tf.square)

    def generate_cube(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            self._common_unary('cube', shape, lambda x: tf.pow(x, 3))

    def generate_pow(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        def pow_der(i, data):
            a, b = data
            if i == 0:
                return b * a ** (b - 1)
            return a ** b * np.log(a)
        for shape in shapes:
            self._common_binary('pow', shape, lambda x, y: x ** y, pow_der)

    def generate_add(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_binary('add', shape,
                lambda x, y: x + y,
                lambda i, data: data1)

    def generate_sub(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        def sub_der(i, data):
            if i == 0:
                return data1
            return -data1
        for shape in shapes:
            data1 = np.ones(shape, dtype=np.float32)
            self._common_binary('sub', shape, lambda x, y: x - y, sub_der)

    def generate_mul(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        def mul_der(i, data):
            if i == 0:
                return data[1]
            return data[0]
        for shape in shapes:
            self._common_binary('mul', shape, lambda x, y: x * y, mul_der)

    def generate_div(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        def div_der(i, data):
            a, b = data
            if i == 0:
                return 1 / b
            return -a / (b * b)
        for shape in shapes:
            self._common_binary('div', shape, lambda x, y: x / y, div_der)

    def generate_min(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        def min_der(i, data):
            a, b = data
            if i == 0:
                return (a <= b).astype(float)
            return (b <= a).astype(float)
        for shape in shapes:
            self._common_binary('min', shape, np.minimum, min_der)

    def generate_max(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        def max_der(i, data):
            a, b = data
            if i == 0:
                return (a >= b).astype(float)
            return (b >= a).astype(float)
        for shape in shapes:
            self._common_binary('max', shape, np.maximum, max_der)

    def generate_eq(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary('eq', shape,
                lambda x, y: np.round(x) == np.round(y),
                lambda i, data: data0)

    def generate_neq(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary('neq', shape,
                lambda x, y: np.round(x) != np.round(y),
                lambda i, data: data0)

    def generate_lt(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary('lt', shape,
                lambda x, y: np.round(x) < np.round(y),
                lambda i, data: data0)

    def generate_gt(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_binary('gt', shape,
                lambda x, y: np.round(x) > np.round(y),
                lambda i, data: data0)

    def generate_nelems(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_unary_raw('nelems', shape,
                lambda data: np.prod(data.shape),
                lambda data: data0)

    def generate_ndims(self):
        shapes = [[3, 4, 5]]
        if 'elementary.shape' in _generate_data:
            shapes += _generate_data['elementary.shape']
        for shape in shapes:
            data0 = np.zeros(shape, dtype=np.float32)
            self._common_unary_raw('ndims', shape,
                lambda data: data.shape[len(shape) - 1],
                lambda data: data0)

    def generate_extend(self):
        self._common_unary_raw('extend', [2],
            lambda data: np.array(list(data) * 3).reshape([3, 2]),
            lambda data: np.array([3, 3]))

    def generate_rsum_1d(self):
        self._common_unary('rsum_1d', [2, 3, 4],
            lambda x: tf.reduce_sum(x, [1]))

    def generate_rprod_1d(self):
        self._common_unary('rprod_1d', [2, 3, 4],
            lambda x: tf.reduce_prod(x, [1]))

    def generate_rmin_1d(self):
        self._common_unary('rmin_1d', [2, 3, 4],
            lambda x: tf.reduce_min(x, [1]))

    def generate_rmax_1d(self):
        self._common_unary('rmax_1d', [2, 3, 4],
            lambda x: tf.reduce_max(x, [1]))

    def generate_rsum(self):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            self._common_unary('rsum', shape, tf.reduce_sum)
            self._common_unary('rsum2', shape,
                lambda x: tf.reduce_sum(x, [0, 1]))

    def generate_rprod(self):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            self._common_unary('rprod', shape, tf.reduce_prod)
            self._common_unary('rprod2', shape,
                lambda x: tf.reduce_prod(x, [0, 1]))

    def generate_rmin(self):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            self._common_unary('rmin', shape, tf.reduce_min)
            self._common_unary('rmin2', shape,
                lambda x: tf.reduce_min(x, [0, 1]))

    def generate_rmax(self):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            self._common_unary('rmax', shape, tf.reduce_max)
            self._common_unary('rmax2', shape,
                lambda x: tf.reduce_max(x, [0, 1]))

    def generate_argmax(self):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            self._common_unary_nograd('argmax', shape, tf.argmax)
            # argmax has no derivatives

    def generate_rl2norm(self):
        shapes = [
            [3, 4, 5],
            [1, 50, 1]
        ]
        for shape in shapes:
            self._common_unary('rl2norm', shape, tf.norm)

    def generate_matmul(self):
        shapes = [
            ([5, 5], [5, 5]),
            ([50, 16], [16, 1]),
            ([2, 5, 6], [2, 6, 7])
        ]

        if 'matmul.shape_shape_left' in _generate_data and\
            'matmul.shape_shape_right' in _generate_data:
            lshape = _generate_data['matmul.shape_shape_left'][0]
            rshape = _generate_data['matmul.shape_shape_right'][0]
            rshape = [lshape[1]] + rshape
            shapes += [(lshape, rshape)]

        for lshape, rshape in shapes:
            is_symmetric = lshape == rshape

            data = np.random.rand(*lshape)
            data2 = np.random.rand(*rshape)

            var = tf.Variable(data)
            var2 = tf.Variable(data2)

            out = tf.matmul(var, var2)
            da, db = tf.gradients(out, [var, var2])

            sess = Session()
            sess.run(var.initializer)
            sess.run(var2.initializer)

            fwd = sess.run(out)
            bwda = sess.run(da)
            bwdb = sess.run(db)

            self.save_case('matmul_fwd', (data, data2), fwd)
            self.save_case('matmul_bwda', (data, data2), bwda)
            self.save_case('matmul_bwdb', (data, data2), bwdb)

            if is_symmetric:
                # test with symmetric property
                tf_both = tf.matmul(var, var)
                dc = tf.gradients(tf_both, [var])[0]

                fboth = sess.run(tf_both)
                bboth = sess.run(dc)
                self.save_case('smatmul_fwd', (data), fboth)
                self.save_case('smatmul_bwd', (data), bboth)

    def generate_convolution(self):
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

            tf_var = tf.Variable(tf_data)
            tf_kernel = tf.Variable(tf_kdata)

            sess = Session()
            sess.run(tf_var.initializer)
            sess.run(tf_kernel.initializer)

            tf_out = tf.nn.convolution(tf_var, tf_kernel, padding='VALID')
            tf_grad, tf_grad2 = tf.gradients(tf_out, [tf_var, tf_kernel])

            fwd = sess.run(tf_out)
            fwd = fwd.reshape([fwd.shape[1], fwd.shape[2]])

            bwd1 = sess.run(tf_grad)
            bwd2 = sess.run(tf_grad2)

            bwd1 = bwd1.reshape(shape)
            bwd2 = bwd2.reshape(kernelshape)

            self.save_case('convolution_fwd', (data, kernel), fwd)
            self.save_case('convolution_dimage', (data, kernel), bwd1)
            self.save_case('convolution_dkernel', (data, kernel), bwd2)

    def generate_conv2d(self):
        # batch, height, width, in
        shape = [3, 3, 3, 2]
        data = np.random.rand(*shape).astype(np.float32)

        # height, width, in, out
        kshape = [2, 2, 2, 4]
        kdata = np.random.rand(*kshape).astype(np.float32)

        tfimage = tf.Variable(data)
        tfkernel = tf.Variable(kdata)

        sess = Session()
        sess.run(tfimage.initializer)
        sess.run(tfkernel.initializer)

        tfoutput = tf.nn.conv2d(tfimage, tfkernel, [1, 1, 1, 1], 'VALID')
        tf_grad, tf_grad2 = tf.gradients(tfoutput, [tfimage, tfkernel])

        fwd = sess.run(tfoutput)
        bwd1 = sess.run(tf_grad)
        bwd2 = sess.run(tf_grad2)

        self.save_case('conv2d_fwd', (data, kdata), fwd)
        self.save_case('conv2d_dimage', (data, kdata), bwd1)
        self.save_case('conv2d_dkernel', (data, kdata), bwd2)

    def generate_avgpool(self):
        # batch, height, width, in
        self._common_unary('avgpool', [3, 8, 8, 2],
            lambda x: tf.nn.avg_pool2d(
                x, [2, 2], [2, 2], padding='VALID'))

    def generate_maxpool(self):
        self._common_unary('maxpool', [3, 8, 8, 2],
            lambda x: tf.nn.max_pool2d(
                x, [2, 2], [2, 2], padding='VALID'))

    def generate_grader_scenario1(self): # REDUCE -> MUL
        data = np.random.rand(3,10)
        data2 = np.random.rand(10)

        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_out = tf.multiply(tf.reduce_sum(tf_var, 0), tf_var2)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess = Session()
        sess.run(tf_var.initializer)
        sess.run(tf_var2.initializer)

        fwd = sess.run(tf_out)
        bwd = sess.run(tf_grad)

        self.save_case('scenario1_fwd', (data, data2), fwd)
        self.save_case('scenario1_bwd', (data, data2), bwd)

    def generate_grader_scenario2(self): # EXTEND -> MUL
        data = np.random.rand(10)
        data2 = np.random.rand(3,10)

        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        tf_out = tf_var * tf_var2
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        sess = Session()
        sess.run(tf_var.initializer)
        sess.run(tf_var2.initializer)

        fwd = sess.run(tf_out)
        bwd = sess.run(tf_grad)

        self.save_case('scenario2_fwd', (data, data2), fwd)
        self.save_case('scenario2_bwd', (data, data2), bwd)

    def generate_grader_scenario3(self): # PERMUTE -> MUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,10)

        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        sess = Session()
        sess.run(tf_var.initializer)
        sess.run(tf_var2.initializer)

        tf_out = tf.transpose(tf_var) * tf_var2
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        fwd = sess.run(tf_out)
        bwd = sess.run(tf_grad)

        self.save_case('scenario3_fwd', (data, data2), fwd)
        self.save_case('scenario3_bwd', (data, data2), bwd)

    def generate_grader_scenario4(self): # MATMUL -> MUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,5)
        data3 = np.random.rand(10,5)

        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_var3 = tf.Variable(data3)

        sess = Session()
        sess.run(tf_var.initializer)
        sess.run(tf_var2.initializer)
        sess.run(tf_var3.initializer)

        tf_out = tf.multiply(tf.matmul(tf_var, tf_var2), tf_var3)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        fwd = sess.run(tf_out)
        bwd = sess.run(tf_grad)

        self.save_case('scenario4_fwd', (data, data2, data3), fwd)
        self.save_case('scenario4_bwd', (data, data2, data3), bwd)

    def generate_grader_scenario5(self): # MATMUL -> MATMUL
        data = np.random.rand(10,3)
        data2 = np.random.rand(3,5)
        data3 = np.random.rand(5,4)

        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)
        tf_var3 = tf.Variable(data3)

        sess = Session()
        sess.run(tf_var.initializer)
        sess.run(tf_var2.initializer)
        sess.run(tf_var3.initializer)

        tf_out = tf.matmul(tf.matmul(tf_var, tf_var2), tf_var3)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        fwd = sess.run(tf_out)
        bwd = sess.run(tf_grad)

        self.save_case('scenario5_fwd', (data, data2, data3), fwd)
        self.save_case('scenario5_bwd', (data, data2, data3), bwd)

    def generate_grader_scenario6(self): # REDUCE -> MATMUL
        data = np.random.rand(4,10,3)
        data2 = np.random.rand(3,5)

        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        sess = Session()
        sess.run(tf_var.initializer)
        sess.run(tf_var2.initializer)

        tf_out = tf.matmul(tf.reduce_sum(tf_var, 0), tf_var2)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        fwd = sess.run(tf_out)
        bwd = sess.run(tf_grad)

        self.save_case('scenario6_fwd', (data, data2), fwd)
        self.save_case('scenario6_bwd', (data, data2), bwd)

    def generate_grader_scenario7(self): # EXTEND -> MATMUL
        data = np.random.rand(3)
        data2 = np.random.rand(3,5)
        ones = np.ones([10, 3])

        tf_var = tf.Variable(data)
        tf_var2 = tf.Variable(data2)

        sess = Session()
        sess.run(tf_var.initializer)
        sess.run(tf_var2.initializer)

        tf_out = tf.matmul(tf_var * ones, tf_var2)
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        fwd = sess.run(tf_out)
        bwd = sess.run(tf_grad)

        self.save_case('scenario7_fwd', (data, data2), fwd)
        self.save_case('scenario7_bwd', (data, data2), bwd)

    # A<a> -> EXTEND -> B<a,b>
    # A<a> -> EXTEND -> C<a,c> -> PERMUTE -> <c,a>
    # B MATMUL C -> D<c,b>
    def generate_grader_scenario8(self):
        data = np.random.rand(5)
        ones = np.ones([10, 5])
        ones2 = np.ones([3, 5])

        tf_var = tf.Variable(data)

        sess = Session()
        sess.run(tf_var.initializer)

        tf_out = tf.matmul(tf_var * ones, tf.transpose(tf_var * ones2))
        tf_grad = tf.gradients(tf_out, [tf_var])[0]

        fwd = sess.run(tf_out)
        bwd = sess.run(tf_grad)

        self.save_case('scenario8_fwd', (data), fwd)
        self.save_case('scenario8_bwd', (data), bwd)

    def generate_grader_scenario9(self):
        ashape = [2, 3]
        bshape = [3, 4]
        cshape = [4, 2]

        data = np.random.rand(*ashape)
        data2 = np.random.rand(*bshape)
        data3 = np.random.rand(*cshape)

        a = tf.Variable(data)
        b = tf.Variable(data2)
        c = tf.Variable(data3)

        d = tf.matmul(a, b)
        e = tf.matmul(c, d)
        f = tf.matmul(tf.transpose(d), tf.transpose(c))
        dest = tf.matmul(e, f)

        da, db, dc = tf.gradients(dest, [a, b, c])

        sess = Session()
        sess.run(a.initializer)
        sess.run(b.initializer)
        sess.run(c.initializer)

        fwd = sess.run(dest)
        da = sess.run(da)
        db = sess.run(db)
        dc = sess.run(dc)

        self.save_case('scenario9_fwd', (data, data2, data3), fwd)
        self.save_case('scenario9_da', (data, data2, data3), da)
        self.save_case('scenario9_db', (data, data2, data3), db)
        self.save_case('scenario9_dc', (data, data2, data3), dc)

if version_lt(tf.__version__, '1.6.0'):
    delattr(TFWriter, 'generate_maxpool')
    delattr(TFWriter, 'generate_avgpool')

if __name__ == "__main__":
    with open('testutil/ead_template.json') as json_data:
        generate_template = json.load(json_data)
        assert 'test_cases' in generate_template
        assert 'config_pools' in generate_template

    # log to file
    logging.basicConfig(filename='/tmp/ead_ptest.log',level=logging.DEBUG)
    logging.info("running tf testdata generation")

    _generate_data = generate_testcases(
        generate_template['test_cases'],
        generate_template['config_pools'])

    writer = TFWriter(sys.argv[1])
    writer.run_all()
    writer.write()
