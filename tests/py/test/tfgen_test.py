''' Test tfgen '''

import string
import random
import unittest

from graphast.gen import allOps
import tenncorgen.tfgen as tfgen
import tenncorgen.utils as utils
import tenncorgen.save_ast as save_ast

def treesize(root):
	def count(_, deps):
		return sum(deps) + 1, None
	counter, _ = utils.traverse(root, count)
	return counter

def randarr(n):
	return [save_ast._randVariable(16) for _ in range(n)]

class TestTfgen(unittest.TestCase):
	def test_tf_gen(self):
		ops = allOps("structure.yml")
		for opname in ops: 
			tops = ops[opname]
			for root in tops:
				mockgid = "gid_" + opname
				createorder = randarr(treesize(root))
				script = tfgen.tf_gen(root, mockgid, createorder)
				compile(script, 'mockfile.py', 'exec')
				self.assertTrue(mockgid in script)
