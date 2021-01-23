import os
import json
import logging
import unittest
import glob

from testutil.generate_testcases import generate_testcases

import test_api
import test_layer
import test_distrib

test_mods = [
    'test_api',
    'test_layer',
    'test_distrib',
]

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

    unittest.main(defaultTest=test_mods)
