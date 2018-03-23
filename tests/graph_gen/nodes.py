#!/usr/bin/env python3

from shapeclass import ELEM, MATMUL, REDUCE, util
import numpy as np

class node:
	def __init__(self, name, sname, depth=0):
		self.name = name
		self.depth = depth
		self.args = []
		if sname == "ELEM":
			self.shaper = ELEM.ELEM
		elif sname == "MATMUL":
			self.shaper = MATMUL.MATMUL
		elif sname == "REDUCE":
			self.shaper = REDUCE.REDUCE
		else:
			raise Exception("unsupported shape class "+sname)

	def __repr__(self):
		return '{"name": %s, "args": [%s]}' % (self.name, ', '.join([str(arg) for arg in self.args]))

	def make_shape(self, shape, value=None):
		self.shaper(shape, [arg.make_shape for arg in self.args])

class leaf:
	def __init__(self):
		self.shape = []

	def __repr__(self):
		return 'LEAF('+str(self.shape)+')'

	def make_shape(self, shape, value=None):
		self.shape = shape

class scalar:
	def __init__(self, dtype):
		if dtype == "double":
			self.value = "random.random() * 37 - 17.4"
		elif dtype == "int":
			self.value = "random.randint(1, 9)"
		else:
			raise Exception("unsupported type "+dtype)

	def __repr__(self):
		return 'SCALAR('+self.value+')'

	def make_shape(self, shape, value=None):
		if value:
			self.value = value
