""" Serializing ast graph as a tensorflow script """

import random
import string

import graphast.nodes as nodes
from tenncorgen.utils import traverse, randVariable

# file 'savedata.py' defined at runtime
tfScript = '''import tensorflow as tf
from tenncorgen.save_data import profile

prof = profile("{0}")
{1}
{2}

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# scalar results
	for label, scalar in {3}:
		prof.save("scalar", label, scalar)

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
'''

def tf_gen(root, graphid, createOrder):
	leaves = []
	scalars = []
	i = 0
	def declare(node, deps):
		id = createOrder[i]
		i = i + 1
		if isinstance(node, nodes.node):
			funcname = tolower(node.name)
			if "rmax" == funcname:
				funcname = "reduce_max"
			elif "rsum" == funcname:
				funcname = "reduce_sum"
			decl = "tf.%s(%s)" % (funcname, ', '.join(deps))
		elif isinstance(node, nodes.leaf):
			decl = "tf.Variable(tf.random_uniform(%s))" % str(node.shape)
			leaves.append(id)
		elif isinstance(node, nodes.scalar):
			decl = node.value
			scalars.append(id)
		else:
			raise Exception("unsupported node type")
		return id, "%s = %s" % (id, decl)

	id, lines = traverse(root, declare)
	grads = ["grad_" + leaf for leaf in leaves]
	tfGrad = "%s = tf.gradients(%s, [%s])" % \
		(', '.join(grads), id, ', '.join(leaves))

	scalarMap = ', '.join([ '"{0}": {0}'.format(scalar) for scalar in scalars])
	leafMap = ', '.join([ '"{0}": {0}'.format(leaf) for leaf in leaves])
	gradMap = ', '.join([ '"{0}": {0}'.format(grad) for grad in grads])

	script = tfScript.format(
		graphid,
		'\n'.join(lines),
		tfGrad,  
		"{" + scalarMap + "}",
		"{" + leafMap + "}",
		"{" + gradMap + "}",
		id)
	return script
