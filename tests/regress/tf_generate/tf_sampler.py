#!/usr/bin/env python3
import tensorflow as tf
import random
import numpy as np
import math

sess = tf.Session()

def stringify(arr):
	if type(arr) == np.float32:
		return str(arr)
	s = '['
	for elem in arr:
		strarr = stringify(elem)
		s = s + strarr + ' '
	return s[:-1] + ']'

def to_str(tensor):
	res = sess.run(tensor)
	return stringify(res)

scalrStr = ""
def register(scalar):
	global scalrStr
	scalrStr = scalrStr + " " + str(scalar)
	return scalar

def clearScalr():
	global scalrStr
	scalrStr = ""

def unar_op(name, op, shape):
	global scalrStr
	a = tf.Variable(tf.random_uniform(shape))
	result = op(a)
	da = tf.gradients(result, [a])[0]
	init = tf.global_variables_initializer()
	sess.run(init)
	if len(scalrStr) > 0:
		scalrStr = scalrStr[1:]
	return name + "," + \
		scalrStr + "," + \
		to_str(a) + "," + \
		to_str(result) + "," + \
		to_str(da)

def bin_elem_op(name, op, shape):
	global scalrStr
	a = tf.Variable(tf.random_uniform(shape))
	b = tf.Variable(tf.random_uniform(shape))
	result = op(a, b)
	da, db = tf.gradients(result, [a, b])
	init = tf.global_variables_initializer()
	sess.run(init)
	if len(scalrStr) > 0:
		scalrStr = scalrStr[1:]
	return name + "," + \
		scalrStr + "," + \
		to_str(a) + "," + \
		to_str(b) + "," + \
		to_str(result) + "," + \
		to_str(da) + "," + \
		to_str(db)

def bin_trans_op(name, op, shape1, shape2):
	global scalrStr
	a = tf.Variable(tf.random_uniform(shape1))
	b = tf.Variable(tf.random_uniform(shape2))
	result = op(a, b)
	da, db = tf.gradients(result, [a, b])
	init = tf.global_variables_initializer()
	sess.run(init)
	if len(scalrStr) > 0:
		scalrStr = scalrStr[1:]
	return name + "," + \
		scalrStr + "," + \
		to_str(a) + "," + \
		to_str(b) + "," + \
		to_str(result) + "," + \
		to_str(da) + "," + \
		to_str(db)

# rank lower than 1 denote random rank 
def rand_shape(rank = 0, nLimit = 10000):
	if rank < 1:
		rank = random.randint(1, 5)
	if rank > 1:
		elemLimit = math.floor(math.log(nLimit, rank))
	else:
		elemLimit = nLimit
	return np.random.randint(math.floor(elemLimit / 2), high=elemLimit, size=rank).astype('int32')
