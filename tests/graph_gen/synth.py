""" Serializing ast graph as a tensorflow script """

import random
import string

import nodes

tfScript = '''import tensorflow as tf
import serialsynth

{0}
{1}

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# scalar results
	for label, scalar in {2}:
		serialsynth.save("scalar", label, scalar)

	# variable results
	for label, input in {3}:
		res = sess.run(input)
		# save to protobuf
		serialsynth.save("variable", label, res)

	# gradient results
	for label, gradres in {4}:
		res = sess.run(gradres)
		serialsynth.save("gradient", label, res)

	# output result
	serialsynth.save("output", "{5}", sess.run({5}))
'''

def randVariable(n):
	postfix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n-1))
	return random.choice(string.ascii_uppercase + string.ascii_lowercase) + postfix

# bottom up traverse tree
def traverse(root, declare):
	lines = []
	deps = []
	if isinstance(root, nodes.node):
		for arg in root.args:
			depid, sublines = traverse(arg, declare)
			lines.extend(sublines)
			deps.append(depid)
	id, decl = declare(root, deps)
	lines.append(decl)
	return id, lines

def synth(root):
	leaves = []
	scalars = []
	def declare(node, deps):
		id = randVariable(16)
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

	id, lines = traverse(root, declare)
	grads = ["grad_" + leaf for leaf in leaves]
	tfGrad = "%s = tf.gradients(%s, [%s])" % \
		(', '.join(grads), id, ', '.join(leaves))

	scalarMap = ', '.join([ '"{0}": {0}'.format(scalar) for scalar in scalars])
	leafMap = ', '.join([ '"{0}": {0}'.format(leaf) for leaf in leaves])
	gradMap = ', '.join([ '"{0}": {0}'.format(grad) for grad in grads])

	script = tfScript.format(
		'\n'.join(lines),
		tfGrad,  
		"{" + scalarMap + "}",
		"{" + leafMap + "}",
		"{" + gradMap + "}",
		id)
	return script
