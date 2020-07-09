import json
import random
import logging
import unittest
import numpy as np
import tensorflow as tf

import tenncor as tc

from testutil.array_testcase import ArrTest
from testutil.tf_testutil import tf_init

test_service = 'tenncor_distrib_ptest'

tfSess = tf_init()

class DISTRIBTest(ArrTest):
    # distributed version of //eteq:ptest
    # test_grader_scenario9 without derivation
    def test_datapassing(self):
        consul = tc.Consul()

        ashape = [2, 3]
        bshape = [3, 4]
        cshape = [4, 2]

        data = np.random.rand(*ashape)
        data2 = np.random.rand(*bshape)
        data3 = np.random.rand(*cshape)

        tf_a = tf.Variable(data)
        tf_b = tf.Variable(data2)
        tf_c = tf.Variable(data3)
        tf_d = tf.matmul(tf_a, tf_b)
        tf_e = tf.matmul(tf_c, tf_d)
        tf_f = tf.matmul(tf.transpose(tf_d), tf.transpose(tf_c))
        tf_root = tf.matmul(tf_e, tf_f)

        tfsess = tfSess()
        tfsess.run(tf_a.initializer)
        tfsess.run(tf_b.initializer)
        tfsess.run(tf_c.initializer)

        exdata = tfsess.run(tf_root)

        def distrib_scope():
            # commons
            a = tc.variable(data, 'a')
            b = tc.variable(data2, 'b')
            c = tc.variable(data3, 'c')

            # sess1
            sess1 = tc.DistribSess(consul, port=5122,
                service_name=test_service, alias='sess1')

            d = tc.api.matmul(a, b)
            f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))

            sess1.track([d, f])
            d_id = sess1.lookup_id(d)
            f_id = sess1.lookup_id(f)

            # sess2
            sess2 = tc.DistribSess(consul, port=5123,
                service_name=test_service, alias='sess2')

            d_ref = sess2.lookup_node(d_id)
            f_ref = sess2.lookup_node(f_id)

            e = tc.api.matmul(c, d_ref)
            root = tc.api.matmul(e, f_ref)

            sess2.track([root])
            sess2.update_target([root])

            return root.get()

        data = distrib_scope()
        self._array_close(exdata, data)

    # distributed version of //eteq:ptest
    # test_grader_scenario9 without derivation
    def test_multipass(self):
        consul = tc.Consul()

        ashape = [2, 3]
        bshape = [3, 4]
        cshape = [4, 2]

        data = np.random.rand(*ashape)
        data2 = np.random.rand(*bshape)
        data3 = np.random.rand(*cshape)

        tf_a = tf.Variable(data)
        tf_b = tf.Variable(data2)
        tf_c = tf.Variable(data3)
        tf_d = tf.matmul(tf_a, tf_b)
        tf_e = tf.matmul(tf_c, tf_d)
        tf_f = tf.matmul(tf.transpose(tf_d), tf.transpose(tf_c))
        tf_root = tf.matmul(tf_e, tf_f)

        tfsess = tfSess()
        tfsess.run(tf_a.initializer)
        tfsess.run(tf_b.initializer)
        tfsess.run(tf_c.initializer)

        exdata = tfsess.run(tf_root)

        def distrib_scope():
            # common
            a = tc.variable(data, 'a')
            b = tc.variable(data2, 'b')
            c = tc.variable(data3, 'c')

            # sess1
            sess1 = tc.DistribSess(consul, port=5122,
                service_name=test_service, alias='sess1')

            d = tc.api.matmul(a, b)

            sess1.track([d])
            d_id = sess1.lookup_id(d)

            # sess2 -> depends on sess1 (reference to d)
            sess2 = tc.DistribSess(consul, port=5123,
                service_name=test_service, alias='sess2')

            d_ref = sess2.lookup_node(d_id)
            f = tc.api.matmul(tc.api.transpose(d_ref), tc.api.transpose(c))

            sess2.track([f])
            f_id = sess2.lookup_id(f)

            # sess3 -> depends on sess1 (reference to d) and sess2 (reference to f)
            sess3 = tc.DistribSess(consul, port=5124,
                service_name=test_service, alias='sess3')

            d_ref2 = sess3.lookup_node(d_id)
            e = tc.api.matmul(c, d_ref2)
            f_ref = sess3.lookup_node(f_id)
            root = tc.api.matmul(e, f_ref)

            sess3.track([root])
            sess3.update_target([root])

            return root.get()

        data = distrib_scope()
        self._array_close(exdata, data)

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
    #         sess1 = tc.DistribSess(consul,
    #             port=5122, service_name=test_service)
    #         sess2 = tc.DistribSess(consul,
    #             port=5123, service_name=test_service)

    #         a = tc.variable(data, 'a')
    #         b = tc.variable(data2, 'b')
    #         c = tc.variable(data3, 'c')

    #         d = tc.api.matmul(a, b)
    #         e = tc.api.matmul(c, d)
    #         f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))
    #         root = tc.api.matmul(e, f)

    #         sess1.track([root])
    #         root_id = sess1.lookup_id(root)
    #         base_id = sess1.lookup_id(a)

    #         root_ref = sess2.lookup_node(root_id)
    #         base_ref = sess2.lookup_node(base_id)
    #         dbase = tc.derive(root_ref, [base_ref])[0]

    #         sess2.track([dbase])
    #         sess2.update_target([dbase])
    #         return dbase.get()

    #     dbase = distrib_scope()
    #     self._array_close(exdata, dbase)

if __name__ == "__main__":
    unittest.main()
