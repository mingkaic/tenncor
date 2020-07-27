import json
import random
import logging
import unittest
import numpy as np

import tenncor as tc

from testutil.array_testcase import ArrTest
from testutil.compare_testcase import TestReader

_test_service = 'tenncor_distrib_ptest'

_testcases = 'models/test/tf_testcases.json'

_reader = TestReader(_testcases)

def _distrib_datapassing(consul, data, data2, data3):
    # commons
    a = tc.variable(data, 'a')
    b = tc.variable(data2, 'b')
    c = tc.variable(data3, 'c')

    # mgr1
    mgr1 = tc.DistManager(consul, port=5122,
        service_name=_test_service, alias='mgr1')

    d = tc.api.matmul(a, b)
    f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))

    mgr1.expose_node(d)
    mgr1.expose_node(f)
    d_id = mgr1.lookup_id(d)
    f_id = mgr1.lookup_id(f)

    # eval2
    mgr2 = tc.DistManager(consul, port=5123,
        service_name=_test_service, alias='mgr2')

    d_ref = mgr2.lookup_node(d_id)
    f_ref = mgr2.lookup_node(f_id)

    e = tc.api.matmul(c, d_ref)
    root = tc.api.matmul(e, f_ref)

    tc.DistEvaluator(mgr2).evaluate([root])

    return root.raw()

def _distrib_multipass(consul, data, data2, data3):
    # common
    a = tc.variable(data, 'a')
    b = tc.variable(data2, 'b')
    c = tc.variable(data3, 'c')

    # mgr1
    mgr1 = tc.DistManager(consul, port=5122,
        service_name=_test_service, alias='mgr1')

    d = tc.api.matmul(a, b)

    mgr1.expose_node(d)
    d_id = mgr1.lookup_id(d)

    # mgr2 -> depends on eval1 (reference to d)
    mgr2 = tc.DistManager(consul, port=5123,
        service_name=_test_service, alias='mgr2')

    d_ref = mgr2.lookup_node(d_id)
    f = tc.api.matmul(tc.api.transpose(d_ref), tc.api.transpose(c))

    mgr2.expose_node(f)
    f_id = mgr2.lookup_id(f)

    # mgr3 -> depends on eval1 (reference to d) and eval2 (reference to f)
    mgr3 = tc.DistManager(consul, port=5124,
        service_name=_test_service, alias='mgr3')

    d_ref2 = mgr3.lookup_node(d_id)
    e = tc.api.matmul(c, d_ref2)
    f_ref = mgr3.lookup_node(f_id)
    root = tc.api.matmul(e, f_ref)

    tc.DistEvaluator(mgr3).evaluate([root])

    return root.raw()

class DISTRIBTest(ArrTest):
    # distributed version of //eteq:ptest
    # test_grader_scenario9 without derivation
    def test_datapassing(self):
        consul = tc.Consul()

        cases = _reader.get_case('scenario9_fwd')
        assert(len(cases) > 0)
        for inps, out in cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)

            res = _distrib_datapassing(consul, data, data2, data3)
            self._array_close(np.array(out), res)

    # distributed version of //eteq:ptest
    # test_grader_scenario9 without derivation
    def test_multipass(self):
        consul = tc.Consul()

        cases = _reader.get_case('scenario9_fwd')
        assert(len(cases) > 0)
        for inps, out in cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)

            res = _distrib_multipass(consul, data, data2, data3)
            self._array_close(np.array(out), res)

    # # derive between distributions
    # def test_crossderivation(self):
    #     consul = tc.Consul()

    #     ashape = [2, 3]
    #     bshape = [3, 4]
    #     cshape = [4, 2]

    #     data = np.random.rand(*ashape)
    #     data2 = np.random.rand(*bshape)
    #     data3 = np.random.rand(*cshape)

    #     tf_a = tf.Variable(data)
    #     tf_b = tf.Variable(data2)
    #     tf_c = tf.Variable(data3)
    #     tf_d = tf.matmul(tf_a, tf_b)
    #     tf_e = tf.matmul(tf_c, tf_d)
    #     tf_f = tf.matmul(tf.transpose(tf_d), tf.transpose(tf_c))
    #     tf_root = tf.matmul(tf_e, tf_f)
    #     tf_dbase = tf.gradients(tf_root, [tf_a])

    #     tfsess = tfSess()
    #     tfsess.run(tf_a.initializer)
    #     tfsess.run(tf_b.initializer)
    #     tfsess.run(tf_c.initializer)

    #     exdata = tfsess.run(tf_dbase)

    #     def distrib_scope():
    #         eval1 = tc.DistrEvaluator(consul,
    #             port=5122, service_name=_test_service)
    #         eval2 = tc.DistrEvaluator(consul,
    #             port=5123, service_name=_test_service)

    #         a = tc.variable(data, 'a')
    #         b = tc.variable(data2, 'b')
    #         c = tc.variable(data3, 'c')

    #         d = tc.api.matmul(a, b)
    #         e = tc.api.matmul(c, d)
    #         f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))
    #         root = tc.api.matmul(e, f)

    #         eval1.expose_node(root)
    #         eval1.expose_node(a)
    #         root_id = eval1.lookup_id(root)
    #         base_id = eval1.lookup_id(a)

    #         root_ref = eval2.lookup_node(root_id)
    #         base_ref = eval2.lookup_node(base_id)
    #         dbase = tc.derive(root_ref, [base_ref])[0]
    #         eval2.evaluate([dbase])

    #         return dbase.raw()

    #     dbase = distrib_scope()
    #     self._array_close(exdata, dbase)

if __name__ == "__main__":
    unittest.main()
