""" Shape propagation for MATMUL class operations """

from shapeclass.util import randint

def MATMUL(parent, makes):
	assert len(makes) == 2
	if len(parent) < 2:
		parent = [parent[0], 1]
	common = randint()
	makes[0]([parent[0], common])
	makes[1]([common, parent[1]])
