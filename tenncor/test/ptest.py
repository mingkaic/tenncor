import json
import random
import logging
import unittest
import numpy as np

import tenncor as tc

from testutil.array_testcase import ArrTest
from testutil.generate_testcases import generate_testcases
from testutil.compare_testcase import TestReader

_testcases = 'models/test/tf_testcases.json'

_test_data = {}

_reader = TestReader(_testcases)

def _round_helper(x):
    if isinstance(x, float):
        return round(x)
    return tc.api.round(x)

class TenncorTest(ArrTest):
    def _common_assign(self, case, api):
        pos_cases = _reader.get_case(case + '_pos')
        assert(len(pos_cases) > 0)
        for inps, out in pos_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)

            var = tc.variable(data, 'var_target')
            src = tc.variable(data2, 'var_source')
            ass1 = api(var, src)

            self._array_close(np.array(out), ass1.get())

        neg_cases = _reader.get_case(case + '_neg')
        assert(len(neg_cases) > 0)
        for inps, out in neg_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)

            var2 = tc.variable(data, 'var_target2')
            src2 = tc.variable(data2, 'var_source2')
            ass2 = api(var2, tc.api.neg(src2))

            self._array_close(np.array(out), ass2.get())

    def _common_unary(self, case, api):
        fwd_cases = _reader.get_case(case + '_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')
            res = api(var)

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case(case + '_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data) = inps
            data = np.array(data)
            shape = data.shape
            var = tc.variable(data, 'var')
            res = api(var)

            var2 = tc.variable(data, 'var2')
            ex, zero = tuple(tc.derive(res, [var, var2]))

            self._array_eq(np.zeros(shape, dtype=np.float32), zero.get())
            self._array_close(np.array(out), ex.get())

    def _common_unary_nograd(self, case, api):
        cases = _reader.get_case(case)
        assert(len(cases) > 0)
        for inps, out in cases:
            (data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')
            res = api(var)

            self._array_close(np.array(out), res.get())

    def _common_binary(self, case, api):
        out_cases = _reader.get_case(case + '_out')
        assert(len(out_cases) > 0)
        for inps, out in out_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')

            res = api(var, var2)

            self._array_close(np.array(out), res.get())

        both_cases = _reader.get_case(case + '_both')
        assert(len(both_cases) > 0)
        for inps, out in both_cases:
            (data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = api(var, var)

            self._array_close(np.array(out), res.get())

        clhs_cases = _reader.get_case(case + '_clhs')
        assert(len(clhs_cases) > 0)
        for inps, out in clhs_cases:
            (data, cst) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = api(var, cst)

            self._array_close(np.array(out), res.get())

        crhs_cases = _reader.get_case(case + '_crhs')
        assert(len(crhs_cases) > 0)
        for inps, out in crhs_cases:
            (cst, data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = api(cst, var)

            self._array_close(np.array(out), res.get())


        outda_cases = _reader.get_case(case + '_outda')
        assert(len(outda_cases) > 0)
        for inps, out in outda_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = api(var, var2)
            zero, ex = tuple(tc.derive(res, [var3, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

        outdb_cases = _reader.get_case(case + '_outdb')
        assert(len(outdb_cases) > 0)
        for inps, out in outdb_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = api(var, var2)
            zero, ex = tuple(tc.derive(res, [var3, var2]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

        bothd_cases = _reader.get_case(case + '_bothd')
        assert(len(bothd_cases) > 0)
        for inps, out in bothd_cases:
            (data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = api(var, var)
            ex = tc.derive(res, [var])[0]

            self._array_close(np.array(out), ex.get())

        clhsd_cases = _reader.get_case(case + '_clhsd')
        assert(len(clhsd_cases) > 0)
        for inps, out in clhsd_cases:
            (data, cst) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = api(var, cst)
            ex = tc.derive(res, [var])[0]

            self._array_close(np.array(out), ex.get())

        crhsd_cases = _reader.get_case(case + '_crhsd')
        assert(len(crhsd_cases) > 0)
        for inps, out in crhsd_cases:
            (cst, data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = api(cst, var)
            ex = tc.derive(res, [var])[0]

            self._array_close(np.array(out), ex.get())

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
        self._common_assign('assign', tc.api.assign)

    def test_assign_add(self):
        self._common_assign('assign_add', tc.api.assign_add)

    def test_assign_sub(self):
        self._common_assign('assign_sub', tc.api.assign_sub)

    def test_assign_mul(self):
        self._common_assign('assign_mul', tc.api.assign_mul)

    def test_assign_div(self):
        self._common_assign('assign_div', tc.api.assign_div)

    def test_abs(self):
        self._common_unary('abs', tc.api.abs)

    def test_neg(self):
        self._common_unary('neg', tc.api.neg)

    def test_sin(self):
        self._common_unary('sin', tc.api.sin)

    def test_cos(self):
        self._common_unary('cos', tc.api.cos)

    def test_tan(self):
        self._common_unary('tan', tc.api.tan)

    def test_exp(self):
        self._common_unary('exp', tc.api.exp)

    def test_log(self):
        self._common_unary('log', tc.api.log)

    def test_sqrt(self):
        self._common_unary('sqrt', tc.api.sqrt)

    def test_round(self):
        self._common_unary('round', tc.api.round)

    def test_sigmoid(self):
        self._common_unary('sigmoid', tc.api.sigmoid)

    def test_tanh(self):
        self._common_unary('tanh', tc.api.tanh)

    def test_clip_by_range(self):
        self._common_unary('clip_by_range',
            lambda x: tc.api.clip_by_range(x, 0.3, 0.6))

    def test_clip_by_l2norm(self):
        self._common_unary('clip_by_l2norm',
            lambda x: tc.api.clip_by_l2norm(x, 5))

    def test_softmax(self):
        self._common_unary('softmax0',
            lambda arr: tc.api.softmax(arr, offset=0, ndims=1))
        self._common_unary('softmax1',
            lambda arr: tc.api.softmax(arr, offset=1, ndims=1))

    def test_relu(self):
        self._common_unary('relu', tc.api.relu)

    def test_square(self):
        self._common_unary('square', tc.api.square)

    def test_cube(self):
        self._common_unary('cube', tc.api.cube)

    def test_pow(self):
        self._common_binary('pow', tc.api.pow)

    def test_add(self):
        self._common_binary('add', tc.api.add)
        self._common_binary('add', lambda a, b: a + b)

    def test_sub(self):
        self._common_binary('sub', tc.api.sub)
        self._common_binary('sub', lambda a, b: a - b)

    def test_mul(self):
        self._common_binary('mul', tc.api.mul)
        self._common_binary('mul', lambda a, b: a * b)

    def test_div(self):
        self._common_binary('div', tc.api.div)
        self._common_binary('div', lambda a, b: a / b)

    def test_min(self):
        self._common_binary('min', tc.api.min)

    def test_max(self):
        self._common_binary('max', tc.api.max)

    def test_eq(self):
        self._common_binary('eq',
            lambda x, y: tc.api.eq(_round_helper(x), _round_helper(y)))
        self._common_binary('eq',
            lambda x, y: _round_helper(x) == _round_helper(y))

    def test_neq(self):
        self._common_binary('neq',
            lambda x, y: tc.api.neq(_round_helper(x), _round_helper(y)))
        self._common_binary('neq',
            lambda x, y: _round_helper(x) != _round_helper(y))

    def test_lt(self):
        self._common_binary('lt',
            lambda x, y: tc.api.lt(_round_helper(x), _round_helper(y)))
        self._common_binary('lt',
            lambda x, y: _round_helper(x) < _round_helper(y))

    def test_gt(self):
        self._common_binary('gt',
            lambda x, y: tc.api.gt(_round_helper(x), _round_helper(y)))
        self._common_binary('gt',
            lambda x, y: _round_helper(x) > _round_helper(y))

    def test_nelems(self):
        self._common_unary('nelems', tc.api.n_elems)

    def test_ndims(self):
        self._common_unary('ndims', lambda x: tc.api.n_dims(x, 0))

    def test_extend(self):
        self._common_unary('extend', lambda x: tc.api.extend(x, 1, [3]))

    def test_rsum_1d(self):
        self._common_unary('rsum_1d',
            lambda x: tc.api.reduce_sum_1d(x, 1))

    def test_rprod_1d(self):
        self._common_unary('rprod_1d',
            lambda x: tc.api.reduce_prod_1d(x, 1))

    def test_rmin_1d(self):
        self._common_unary('rmin_1d',
            lambda x: tc.api.reduce_min_1d(x, 1))

    def test_rmax_1d(self):
        self._common_unary('rmax_1d',
            lambda x: tc.api.reduce_max_1d(x, 1))

    def test_rsum(self):
        self._common_unary('rsum', tc.api.reduce_sum)
        self._common_unary('rsum2',
            lambda x: tc.api.reduce_sum(x, offset=1))

    def test_rprod(self):
        self._common_unary('rprod', tc.api.reduce_prod)
        self._common_unary('rprod2',
            lambda x: tc.api.reduce_prod(x, offset=1))

    def test_rmin(self):
        self._common_unary('rmin', tc.api.reduce_min)
        self._common_unary('rmin2',
            lambda x: tc.api.reduce_min(x, offset=1))

    def test_rmax(self):
        self._common_unary('rmax', tc.api.reduce_max)
        self._common_unary('rmax2',
            lambda x: tc.api.reduce_max(x, offset=1))

    def test_argmax(self):
        self._common_unary_nograd('argmax',
            lambda x: tc.api.permute(tc.api.argmax(
                x, return_dim=1), [0, 2, 1]))

    def test_rl2norm(self):
        self._common_unary('rl2norm', tc.api.reduce_l2norm)

    def test_matmul(self):
        fwd_cases = _reader.get_case('matmul_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')

            res = tc.api.matmul(var, var2)

            self._array_close(np.array(out), res.get())

        bwda_cases = _reader.get_case('matmul_bwda')
        assert(len(bwda_cases) > 0)
        for inps, out in bwda_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.matmul(var, var2)
            zero, grad = tuple(tc.derive(res, [var3, var]))

            self._array_close(np.array(out), grad.get())
            self._array_close(data0, zero.get())

        bwdb_cases = _reader.get_case('matmul_bwdb')
        assert(len(bwdb_cases) > 0)
        for inps, out in bwdb_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.matmul(var, var2)
            zero, grad = tuple(tc.derive(res, [var3, var2]))

            self._array_close(np.array(out), grad.get())
            self._array_close(data0, zero.get())

        sfwd_cases = _reader.get_case('smatmul_fwd')
        for inps, out in sfwd_cases:
            (data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = tc.api.matmul(var, var)

            self._array_close(np.array(out), res.get())

        sbwd_cases = _reader.get_case('smatmul_bwd')
        for inps, out in sbwd_cases:
            (data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = tc.api.matmul(var, var)
            grad = tuple(tc.derive(res, [var]))[0]

            self._array_close(np.array(out), grad.get())

    def test_convolution(self):
        fwd_cases = _reader.get_case('convolution_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            vkernel = tc.variable(data2, 'vkernel')

            res = tc.api.convolution(var, vkernel, list(range(8)))

            self._array_close(np.array(out), res.get())

        bwda_cases = _reader.get_case('convolution_dimage')
        assert(len(bwda_cases) > 0)
        for inps, out in bwda_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            vkernel = tc.variable(data2, 'vkernel')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.convolution(var, vkernel, list(range(8)))
            zero, grad = tuple(tc.derive(res, [var3, var]))

            self._array_close(np.array(out), grad.get())
            self._array_close(data0, zero.get())

        bwdb_cases = _reader.get_case('convolution_dkernel')
        assert(len(bwdb_cases) > 0)
        for inps, out in bwdb_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            vkernel = tc.variable(data2, 'vkernel')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.convolution(var, vkernel, list(range(8)))
            zero, grad = tuple(tc.derive(res, [var3, vkernel]))

            self._array_close(np.array(out), grad.get())
            self._array_close(data0, zero.get())

    def test_conv2d(self):
        fwd_cases = _reader.get_case('conv2d_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            image = tc.variable(data, 'image')
            kernel = tc.variable(data2, 'kernel')

            res = tc.api.nn.conv2d(image, kernel)

            self._array_close(np.array(out), res.get())

        bwda_cases = _reader.get_case('conv2d_dimage')
        assert(len(bwda_cases) > 0)
        for inps, out in bwda_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            image = tc.variable(data, 'image')
            kernel = tc.variable(data2, 'kernel')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.nn.conv2d(image, kernel)
            zero, grad = tuple(tc.derive(res, [var3, image]))

            self._array_close(np.array(out), grad.get())
            self._array_close(data0, zero.get())

        bwdb_cases = _reader.get_case('conv2d_dkernel')
        assert(len(bwdb_cases) > 0)
        for inps, out in bwdb_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            image = tc.variable(data, 'image')
            kernel = tc.variable(data2, 'kernel')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.nn.conv2d(image, kernel)
            zero, grad = tuple(tc.derive(res, [var3, kernel]))

            self._array_close(np.array(out), grad.get())
            self._array_close(data0, zero.get())

    def test_stride(self):
        shape = [3, 8, 8, 2]
        data = np.random.rand(*shape).astype(np.float32)
        image = tc.variable(data, 'image')

        strideout = tc.api.stride(image, [1, 2, 2])
        self._array_eq([3, 4, 4, 2], strideout.shape())

        ex = tc.derive(strideout, [image])[0]
        self._array_eq(shape, ex.shape())

    def test_avgpool(self):
        self._common_unary('avgpool',
            lambda x: tc.api.nn.mean_pool2d(x, [1, 2]))

    def test_maxpool(self):
        self._common_unary('maxpool',
            lambda x: tc.api.nn.max_pool2d(x, [1, 2]))

    def test_grader_scenario1(self): # REDUCE -> MUL
        fwd_cases = _reader.get_case('scenario1_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')

            res = tc.api.mul(tc.api.reduce_sum(var, offset=1, ndims=1), var2)

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case('scenario1_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.mul(tc.api.reduce_sum(var, offset=1, ndims=1), var2)
            zero, ex = tuple(tc.derive(res, [var3, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

    def test_grader_scenario2(self): # EXTEND -> MUL
        fwd_cases = _reader.get_case('scenario2_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')

            res = tc.api.mul(tc.api.extend(var, 1, [3]), var2)

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case('scenario2_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.mul(tc.api.extend(var, 1, [3]), var2)
            zero, ex = tuple(tc.derive(res, [var3, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

    def test_grader_scenario3(self): # PERMUTE -> MUL
        fwd_cases = _reader.get_case('scenario3_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')

            res = tc.api.mul(tc.api.permute(var, [1,0]), var2)

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case('scenario3_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.mul(tc.api.permute(var, [1,0]), var2)
            zero, ex = tuple(tc.derive(res, [var3, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

    def test_grader_scenario4(self): # MATMUL -> MUL
        fwd_cases = _reader.get_case('scenario4_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data3, 'var3')

            res = tc.api.mul(tc.api.matmul(var, var2), var3)

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case('scenario4_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data3, 'var3')
            var4 = tc.variable(data, 'var4')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.mul(tc.api.matmul(var, var2), var3)
            zero, ex = tuple(tc.derive(res, [var4, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

    def test_grader_scenario5(self): # MATMUL -> MATMUL
        fwd_cases = _reader.get_case('scenario5_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data3, 'var3')

            res = tc.api.matmul(tc.api.matmul(var, var2), var3)

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case('scenario5_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data3, 'var3')
            var4 = tc.variable(data, 'var4')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.matmul(tc.api.matmul(var, var2), var3)
            zero, ex = tuple(tc.derive(res, [var4, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

    def test_grader_scenario6(self): # REDUCE -> MATMUL
        fwd_cases = _reader.get_case('scenario6_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')

            res = tc.api.matmul(tc.api.reduce_sum(var, offset=2, ndims=1), var2)

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case('scenario6_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.matmul(tc.api.reduce_sum(var, offset=2, ndims=1), var2)
            zero, ex = tuple(tc.derive(res, [var3, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

    def test_grader_scenario7(self): # EXTEND -> MATMUL
        fwd_cases = _reader.get_case('scenario7_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')

            res = tc.api.matmul(tc.api.extend(var, 1, [10]), var2)

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case('scenario7_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data, data2) = inps
            data = np.array(data)
            data2 = np.array(data2)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data2, 'var2')
            var3 = tc.variable(data, 'var3')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.matmul(tc.api.extend(var, 1, [10]), var2)
            zero, ex = tuple(tc.derive(res, [var3, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

    # A<a> -> EXTEND -> B<a,b>
    # A<a> -> EXTEND -> C<a,c> -> PERMUTE -> <c,a>
    # B MATMUL C -> D<c,b>
    def test_grader_scenario8(self):
        fwd_cases = _reader.get_case('scenario8_fwd')
        assert(len(fwd_cases) > 0)
        for inps, out in fwd_cases:
            (data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')

            res = tc.api.matmul(
                tc.api.extend(var, 1, [10]),
                tc.api.permute(tc.api.extend(var, 1, [3]), [1, 0]))

            self._array_close(np.array(out), res.get())

        bwd_cases = _reader.get_case('scenario8_bwd')
        assert(len(bwd_cases) > 0)
        for inps, out in bwd_cases:
            (data) = inps
            data = np.array(data)
            var = tc.variable(data, 'var')
            var2 = tc.variable(data, 'var2')

            data0 = np.zeros(data.shape, dtype=np.float32)
            res = tc.api.matmul(
                tc.api.extend(var, 1, [10]),
                tc.api.permute(tc.api.extend(var, 1, [3]), [1, 0]))
            zero, ex = tuple(tc.derive(res, [var2, var]))

            self._array_close(np.array(out), ex.get())
            self._array_eq(data0, zero.get())

    def test_grader_scenario9(self):
        da_cases = _reader.get_case('scenario9_da')
        assert(len(da_cases) > 0)
        for inps, out in da_cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)
            a = tc.variable(data, 'a')
            b = tc.variable(data2, 'b')
            c = tc.variable(data3, 'c')

            d = tc.api.matmul(a, b)
            e = tc.api.matmul(c, d)
            f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))
            res = tc.api.matmul(e, f)
            ex = tc.derive(res, [a])[0]

            self._array_close(np.array(out), ex.get())

        db_cases = _reader.get_case('scenario9_db')
        assert(len(db_cases) > 0)
        for inps, out in db_cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)
            a = tc.variable(data, 'a')
            b = tc.variable(data2, 'b')
            c = tc.variable(data3, 'c')
            g = tc.variable(data, 'g')

            d = tc.api.matmul(a, b)
            e = tc.api.matmul(c, d)
            f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))
            res = tc.api.matmul(e, f)
            ex = tc.derive(res, [b])[0]

            self._array_close(np.array(out), ex.get())

        dc_cases = _reader.get_case('scenario9_dc')
        assert(len(dc_cases) > 0)
        for inps, out in dc_cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)
            a = tc.variable(data, 'a')
            b = tc.variable(data2, 'b')
            c = tc.variable(data3, 'c')
            g = tc.variable(data, 'g')

            d = tc.api.matmul(a, b)
            e = tc.api.matmul(c, d)
            f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))
            res = tc.api.matmul(e, f)
            ex = tc.derive(res, [c])[0]

            self._array_close(np.array(out), ex.get())

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
