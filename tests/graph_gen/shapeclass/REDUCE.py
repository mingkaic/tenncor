""" Shape propagation for REDUCE class operations """

from shapeclass.util import randint, randshape

def REDUCE(parent, makes):
	if len(makes) > 1:
		limit = len(parent)+1
		idx = randint(0, limit)
		mul = randint()
		makes[0](expand(parent, idx, mul))
		makes[1]([1], str(idx))
	else:
		makes[0](randshape())

def expand(shape, idx, mul):
	after = shape[idx:]
	out = shape[:idx]
	out.append(mul)
	out.extend(after)
	return out
