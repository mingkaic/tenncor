import json
import random
import logging
import unittest
import numpy as np
import tempfile
import os.path
from glob import glob
import tempfile

import tenncor as tc

from dbg.print import graph_to_str

from testutil.array_testcase import ArrTest
from testutil.generate_testcases import generate_testcases
from testutil.compare_testcase import joint_json

_test_service = 'tenncor_distrib_ptest'

_testcases = glob('models/test/testcases/tf_cases_*.json')

_test_data = {}

_reader = joint_json(_testcases)

def _distrib_datapassing(consul, data, data2, data3):
    # mgr1
    mgr1 = tc.DistrManager(consul, port=5122,
        service_name=_test_service, alias='mgr1')

    a = tc.variable(data, 'a')
    b = tc.variable(data2, 'b')
    c = tc.variable(data3, 'c')

    d = tc.api.matmul(a, b)
    f = tc.api.matmul(tc.api.transpose(d), tc.api.transpose(c))

    tc.expose_node(d)
    tc.expose_node(f)

    # eval2
    tc.DistrManager(consul, port=5123,
        service_name=_test_service, alias='mgr2')

    d_ref = tc.lookup_node(mgr1.lookup_id(d))
    f_ref = tc.lookup_node(mgr1.lookup_id(f))

    e = tc.api.matmul(c, d_ref)
    root = tc.api.matmul(e, f_ref)

    return root.get()

def _distrib_multipass(consul, data, data2, data3):
    # mgr1
    mgr1 = tc.DistrManager(consul, port=5122,
        service_name=_test_service, alias='mgr1') # keep alive

    a = tc.variable(data, 'a')
    b = tc.variable(data2, 'b')

    d = tc.api.matmul(a, b)

    tc.expose_node(d)
    d_id = tc.lookup_id(d)

    # mgr2 -> depends on eval1 (reference to d)
    mgr2 = tc.DistrManager(consul, port=5123,
        service_name=_test_service, alias='mgr2')

    c = tc.variable(data3, 'c')
    d_ref2 = tc.lookup_node(d_id)

    f = tc.api.matmul(tc.api.transpose(d_ref2), tc.api.transpose(c))

    tc.expose_node(f)
    tc.expose_node(c)

    # mgr3 -> depends on eval1 (reference to d) and eval2 (reference to f)
    tc.DistrManager(consul, port=5124,
        service_name=_test_service, alias='mgr3')

    d_ref3 = tc.lookup_node(d_id)
    c_ref = tc.lookup_node(mgr2.lookup_id(c))
    f_ref = tc.lookup_node(mgr2.lookup_id(f))

    e = tc.api.matmul(c_ref, d_ref3)
    root = tc.api.matmul(e, f_ref)

    return root.get()

def _distrib_crossderive(consul, data, data2, data3):
    # mgr1
    mgr1 = tc.DistrManager(consul, port=5122,
        service_name=_test_service, alias='mgr1')

    a = tc.variable(data, 'a')
    b = tc.variable(data2, 'b')

    d = tc.api.matmul(a, b)

    tc.expose_node(a)
    tc.expose_node(d)
    d_id = tc.lookup_id(d)

    # mgr2 -> depends on eval1 (reference to d)
    mgr2 = tc.DistrManager(consul, port=5123,
        service_name=_test_service, alias='mgr2')

    c = tc.variable(data3, 'c')
    d_ref2 = tc.lookup_node(d_id)

    f = tc.api.matmul(tc.api.transpose(d_ref2), tc.api.transpose(c))

    tc.expose_node(f)
    tc.expose_node(c)

    # mgr3 -> depends on eval1 (reference to d) and eval2 (reference to f)
    mgr3 = tc.DistrManager(consul, port=5124,
        service_name=_test_service, alias='mgr3')

    d_ref3 = tc.lookup_node(d_id)
    c_ref = tc.lookup_node(mgr2.lookup_id(c))
    f_ref = tc.lookup_node(mgr2.lookup_id(f))

    e = tc.api.matmul(c_ref, d_ref3)
    root = tc.api.matmul(e, f_ref)

    a_ref = tc.lookup_node(mgr1.lookup_id(a))
    dbase = tc.derive(root, [a_ref])[0]
    # mgr3.print_ascii(dbase)

    return dbase.get()

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

    # derive between distributions
    def test_crossderivation(self):
        consul = tc.Consul()

        cases = _reader.get_case('scenario9_da')
        assert(len(cases) > 0)
        for inps, out in cases:
            (data, data2, data3) = inps
            data = np.array(data)
            data2 = np.array(data2)
            data3 = np.array(data3)

            res = _distrib_crossderive(consul, data, data2, data3)
            self._array_close(np.array(out), res)

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
