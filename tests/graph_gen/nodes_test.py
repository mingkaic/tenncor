""" Test nodes """

import unittest

import re
from nodes import node, leaf, scalar

class mock_leaf:
	def __init__(self):
		self.GSHAPE = []
		self.GVAL = 0

	def make_shape(self, shape, value=None):
		self.GSHAPE = shape
		self.GVAL = value

class TestNode(unittest.TestCase):
	def test_node(self):
		n1 = node("example1", "ELEM")
		n2 = node("example2", "MATMUL", 1)
		n3 = node("example3", "REDUCE", 2)
		self.assertEqual("example1", n1.name)
		self.assertEqual("example2", n2.name)
		self.assertEqual("example3", n3.name)
		self.assertEqual(0, n1.depth)
		self.assertEqual(1, n2.depth)
		self.assertEqual(2, n3.depth)
		self.assertEqual(0, len(n1.args))
		self.assertEqual(0, len(n2.args))
		self.assertEqual(0, len(n3.args))
		inshape = [1, 5, 3, 4, 2]
		l1 = mock_leaf()
		l2 = mock_leaf()
		n1.args = [l1, l2]
		n1.make_shape(inshape)
		self.assertEqual(inshape, l1.GSHAPE)
		self.assertEqual(inshape, l2.GSHAPE)
		self.assertEqual(None, l1.GVAL)
		self.assertEqual(None, l2.GVAL)

		n2.args = [l1, l2]
		n2.make_shape(inshape)
		self.assertEqual(inshape[0], l1.GSHAPE[0])
		self.assertEqual(inshape[1], l2.GSHAPE[1])
		self.assertEqual(l1.GSHAPE[1], l2.GSHAPE[0])
		self.assertEqual(None, l1.GVAL)
		self.assertEqual(None, l2.GVAL)

		n3.args = [l1, l2]
		n3.make_shape(inshape)
		self.assertEqual([1], l2.GSHAPE)
		dim = int(l2.GVAL)
		self.assertTrue(0 <= dim or dim <= len(inshape))
		got = l1.GSHAPE[:min(dim, len(inshape))]
		got.extend(l1.GSHAPE[dim+1:])
		self.assertEqual(inshape, got)
		self.assertEqual(None, l1.GVAL)

	def test_leaf(self):
		l = leaf()
		self.assertEqual([], l.shape)
		inshape = [1, 5, 3, 4, 2]
		l.make_shape(inshape)
		self.assertEqual(inshape, l.shape)

	def test_scalar(self):
		l1 = scalar("double")
		l2 = scalar("int")
		self.assertEqual(l1.value, "random.random() * 37 - 17.4")
		self.assertEqual(l2.value, "random.randint(1, 9)")
		s = 14.2
		l2.make_shape([], s)
		self.assertEqual(s, l2.value)
