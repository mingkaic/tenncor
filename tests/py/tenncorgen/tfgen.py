""" Serializing ast graph as a tensorflow script """

import random
import string

import graphast.nodes as nodes
from tenncorgen.utils import traverse

# file 'save_data.py' defined at runtime
tfScript = '''import sys
import os
import random
if os.path.exists("{7}"):
	sys.path.append("{7}")

import tensorflow as tf
from tests.py.tenncorgen.save_data import profile

prof = profile()
{1}
{2}

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	print("saving random scalar results")
	placemap = {3}
	for label in placemap:
		place = placemap[label]
		# save to protobuf
		prof.save("place", label, place)

	print("saving variable results")
	varmap = {4}
	for label in varmap:
		input = varmap[label]
		res = sess.run(input)
		res = res.astype(float)
		# save to protobuf
		prof.save("variable", label, res)

	print("saving gradient results")
	gradmap = {5}
	for label in gradmap:
		gradres = gradmap[label]
		res = sess.run(gradres)
		res = res.astype(float)
		# save to protobuf
		prof.save("gradient", label, res)

	print("saving output results")
	prof.save("output", "{6}", sess.run({6}).astype(float))

dest = "{0}" + sys.argv[1] + ".data"
print("serializing to " + dest)
prof.serialize(dest)
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
	if len(decl.leaves) > 1:
		tfGrad = "%s = tf.gradients(%s, [%s])" % \
			(', '.join(grads), id, ', '.join(decl.leaves))
	else:
		tfGrad = "%s = tf.gradients(%s, %s)[0]" % \
			(', '.join(grads), id, decl.leaves[0])

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
