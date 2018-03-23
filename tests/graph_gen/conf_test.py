""" Test structure conf """

import unittest

from conf import getOpts

class TestStructConf(unittest.TestCase):
	def test_conf(self):
		opts = getOpts("structure.yml")
		self.assertTrue(isinstance(opts, list))
		expectedKeys = ["class", "shapeclass", "args", "names"]
		for cs in opts:
			self.assertTrue(set(expectedKeys) & set(cs.keys()))
