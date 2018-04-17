""" Serializing ast graph as a tensorflow script """

import random
import string

import graphast.nodes as nodes
from tenncorgen.utils import traverse

# file 'save_data.py' defined at runtime
tfScript = '''import sys
import os
if os.path.exists({7}):
	sys.path.insert(0, {7})

import tensorflow as tf
from tenncorgen.save_data import profile

prof = profile()
{1}
{2}

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# scalar results
	for label, place in {3}:
		prof.save("place", label, place)

	# variable results
	for label, input in {4}:
		res = sess.run(input)
		# save to protobuf
		prof.save("variable", label, res)

	# gradient results
	for label, gradres in {5}:
		res = sess.run(gradres)
		prof.save("gradient", label, res)

	# output result
	prof.save("output", "{6}", sess.run({6}))

prof.serialize("{0}" + sys.argv[1] + ".data")
'''

class declarable:
	def __init__(self, createOrder):
		self.createOrder = createOrder
		self.leaves = []
		self.placescalars = []
		self.i = 0

	def nextId(self):
		id = self.createOrder[self.i]
		self.i = self.i + 1
		return id

	def declare(self, node, deps):
		id = self.nextId()
		if isinstance(node, nodes.node):
			funcname = node.name.lower()
			if "rmax" == funcname:
				funcname = "reduce_max"
			elif "rsum" == funcname:
				funcname = "reduce_sum"
			decl = "tf.%s(%s)" % (funcname, ', '.join(deps))
		elif isinstance(node, nodes.leaf):
			decl = "tf.Variable(tf.random_uniform(%s))" % str(node.shape)
			self.leaves.append(id)
		elif isinstance(node, nodes.scalar):
			decl = node.value
			try:
				int(node.value)
			except ValueError:
				self.placescalars.append(id)
		else:
			raise Exception("unsupported node type")
		return id, "%s = %s" % (id, decl)

def tf_gen(root, graphid, createOrder, out_prefix = "", external = ""):
	decl = declarable(createOrder)
	id, lines = traverse(root, decl.declare)
	grads = ["grad_" + leaf for leaf in decl.leaves]
	tfGrad = "%s = tf.gradients(%s, [%s])" % \
		(', '.join(grads), id, ', '.join(decl.leaves))

	placeMap = ', '.join([ '"{0}": {0}'.format(place) for place in decl.placescalars])
	leafMap = ', '.join([ '"{0}": {0}'.format(leaf) for leaf in decl.leaves])
	gradMap = ', '.join([ '"{0}": grad_{0}'.format(leaf) for leaf in decl.leaves])

	script = tfScript.format(
		out_prefix + graphid,
		'\n'.join(lines),
		tfGrad,
		"{" + placeMap + "}",
		"{" + leafMap + "}",
		"{" + gradMap + "}",
		id, external)
	return script
