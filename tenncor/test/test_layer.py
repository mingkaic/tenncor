import json
import random
import logging
import unittest
import numpy as np
import tempfile
import os.path
from glob import glob

import tenncor as tc

from dbg.print import graph_to_str

from testutil.array_testcase import ArrTest
from testutil.generate_testcases import generate_testcases
from testutil.compare_testcase import joint_json

_test_service = 'tenncor_distrib_ptest'

_testcases = glob('models/test/testcases/tf_cases_*.json')

_test_data = {}

_reader = joint_json(_testcases)

class LAYRTest(unittest.TestCase):
    def test_context_save(self):
        context = tc.Context()

        nunits = 9
        ninput = 10
        noutput = int(ninput / 2)
        n_batch = 10

        api = tc.TenncorAPI(context)

        train_err = tc.apply_update([api.layer.link([
                api.layer.dense([ninput], [nunits],
                    kernel_init=api.init.xavier_uniform(),
                    bias_init=api.init.zeros()),
                api.layer.bind(api.sigmoid),
                api.layer.dense([nunits], [noutput],
                    kernel_init=api.init.xavier_uniform(),
                    bias_init=api.init.zeros()),
                api.layer.bind(api.sigmoid),
            ])],
            lambda err, vars: api.approx.sgd(err, vars, learning_rate=0.9),
            lambda models: api.loss.mean_squared(
                tc.EVariable([n_batch, noutput]), models[0].connect(
                    tc.EVariable([n_batch, ninput]))),
            ctx=context)
        self.assertEqual(1, len(context.get_actives()))
        err_str = graph_to_str(train_err)

        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, 'layr_test')
        tc.save_context_file(test_file, context)

        context2 = tc.Context()
        roots = tc.load_context_file(test_file, context2)
        self.assertEqual(1, len(roots))
        root_str = graph_to_str(list(roots)[0])
        self.assertEqual(err_str, root_str)

    def test_gru(self):
        brain = tc.api.layer.link([
            tc.api.layer.conv2d([5, 5], 1, 16,
                kernel_init=tc.api.init.xavier_normal(0.5),
                zero_padding=((2, 2), (2, 2))), # input of [1, 2641, 128, 1] -> output of [16, 2647, 128, 1]
            tc.api.layer.conv2d([5, 5], 16, 20,
                kernel_init=tc.api.init.xavier_normal(0.5),
                zero_padding=((2, 2), (2, 2))), # input of [16, 2647, 128, 1] -> output of [20, 2647, 128, 1]
            tc.api.layer.gru(tc.Shape([2647, 20]), 20, 128,
                seq_dim = 2,
                kernel_init=tc.api.init.zeros(),
                bias_init=tc.api.init.zeros()), # input of [20, 2647, 128, 1] -> output of [20, 2647, 128]
        ], tc.EVariable([128, 2647, 1], label='input'))

        # state = [x, 2647]
        # concat([20, 2647] [x, 2647]) -> [20 + x, 2647]
        # inshape = [20 + x, 2647]
        # [20 + x, 2647] @ [x, 20 + x] -> [x, 2647]
        self.assertEqual([128, 2647, 20], list(brain.shape()))

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
