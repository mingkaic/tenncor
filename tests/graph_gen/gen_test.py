""" Test random equation ast """

import unittest

import gen
import nodes

class TestAst(unittest.TestCase):
	def test_rand(self):
		emindepth = 2
		emaxdepth = 5
		g = gen.generator("structure.yml", emindepth, emaxdepth)
		root = g.generate()
		depth = 0
		self.assertEqual(0, root.depth)
		stacks = [root]
		while len(stacks) > 0:
			iter = stacks.pop()
			for arg in iter.args:
				if isinstance(arg, nodes.node):
					self.assertEqual(iter.depth+1, arg.depth)
					stacks.append(arg)
			depth = iter.depth
		self.assertTrue(emindepth <= depth and depth <= emaxdepth)

	def test_all(self):
		ops = gen.allOps("structure.yml")
		for opname in ops:
			tops = ops[opname]
			for res in tops:
				self.assertEqual(opname, res.name)
				for arg in res.args:
					self.assertTrue(isinstance(arg, nodes.scalar) or isinstance(arg, nodes.leaf))
