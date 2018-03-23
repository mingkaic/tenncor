""" Synthesis exectuable """

import random
import string
from gen import generator
import nodes

tfScript = '''import tensorflow as tf
import serialsynth

{0}
{1}

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for label, input in {2}:
		res = sess.run(input)
		# save to protobuf
		serialsynth.save(label, res)

	for label, result in {3}:
		res = sess.run(result)
		serialsynth.save(label, res)
	
	for label, scalar in {4}:
		serialsynth.save(label, scalar)
'''

MINDEPTH = os.environ['MINDEPTH'] if 'MINDEPTH' in os.environ else 1
MAXDEPTH = os.environ['MAXDEPTH'] if 'MAXDEPTH' in os.environ else 10

def randVariable(n):
	postfix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n-1))
	return random.choice(string.ascii_uppercase + string.ascii_lowercase) + postfix

# bottom up traverse tree
def traverse(root, declare):
	lines = []
	deps = []
	for arg in root.args:
		depid, sublines = traverse(arg, declare)
		lines.extend(sublines)
		deps.append(depid)
	id, decl = declare(root, deps)
	lines.append(decl)
	return id, lines

def tfGen(root):
	leaves = []
	scalars = []
	def declare(node, deps):
		id = randVariable(16)
		if isinstance(node, nodes.node):
			decl = "tf.%s(%s)" % node.name, deps.join(", ")
		elif isinstance(node, nodes.leaf):
			decl = "tf.Variable(%s)" %s str(nodes.shape)
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

	leafMap = ', '.join([ '"{0}": {0}'.format(leaf) for leaf in leaves])
	gradMap = ', '.join([ '"{0}": {0}'.format(grad) for grad in grads])
	scalarMap = ', '.join([ '"{0}": {0}'.format(scalar) for scalar in scalars])
	script = tfScript.format('\n'.join(lines), 
		tfGrad, "{" + leafMap + "}", 
		"{" + id + ", " + gradMap + "}", 
		"{" + scalarMap + "}")
	print(script)

def main():
	rgen = generator("structure.yml", MINDEPTH, MAXDEPTH)
	root = rgen.generate()
	tfGen(root)

if __name__ == "__main__":
	main()
