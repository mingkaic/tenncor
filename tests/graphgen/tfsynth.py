""" Serializing ast graph as a tensorflow script """

import random
import string

import graphast.nodes as nodes

# file 'savedata.py' defined at runtime
tfScript = '''import tensorflow as tf
from savedata import profile

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

def _randVariable(n):
	postfix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n-1))
	return random.choice(string.ascii_uppercase + string.ascii_lowercase) + postfix

# bottom up traverse tree
def _traverse(root, declare):
	lines = []
	deps = []
	if isinstance(root, nodes.node):
		for arg in root.args:
			depid, sublines = _traverse(arg, declare)
			lines.extend(sublines)
			deps.append(depid)
	id, decl = declare(root, deps)
	lines.append(decl)
	return id, lines

def tfsynth(root):
	leaves = []
	scalars = []
	def declare(node, deps):
		id = _randVariable(16)
		if isinstance(node, nodes.node):
			decl = "tf.%s(%s)" % (node.name, ', '.join(deps))
		elif isinstance(node, nodes.leaf):
			decl = "tf.Variable(tf.random_uniform(%s))" % str(node.shape)
			leaves.append(id)
		elif isinstance(node, nodes.scalar):
			decl = node.value
			scalars.append(id)
		else:
			raise Exception("supportede node type")
		return id, "%s = %s" % (id, decl)

	id, lines = _traverse(root, declare)
	grads = ["grad_" + leaf for leaf in leaves]
	tfGrad = "%s = tf.gradients(%s, [%s])" % \
		(', '.join(grads), id, ', '.join(leaves))

	scalarMap = ', '.join([ '"{0}": {0}'.format(scalar) for scalar in scalars])
	leafMap = ', '.join([ '"{0}": {0}'.format(leaf) for leaf in leaves])
	gradMap = ', '.join([ '"{0}": {0}'.format(grad) for grad in grads])

	graphid = _randVariable(32)
	script = tfScript.format(
		graphid,
		'\n'.join(lines),
		tfGrad,  
		"{" + scalarMap + "}",
		"{" + leafMap + "}",
		"{" + gradMap + "}",
		id)
	return script
