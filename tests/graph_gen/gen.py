#!/usr/bin/env python3

import os
import math
import random
import conf
import re
from shapeclass.util import randshape
import nodes

scalarPattern = re.compile(r'scalar\((.*)\)')

class template:
	def __init__(self, gen, sname, atypes):
		self.gen = gen
		self.sname = sname
		self.atypes = atypes

	def getNexts(self, iter):
		if iter.depth < self.gen.mindepth:
			prTerm = 0.0
		else:
			prTerm = math.sqrt(float(iter.depth - self.gen.mindepth) / self.gen.depthdiff)
		atypes = random.choice(self.atypes)
		for argtype in atypes:
			scalarM = scalarPattern.search(argtype)
			if scalarM:
				dtype = scalarM.group(1)
				next = nodes.scalar(dtype)
			elif argtype == "tensor":
				if random.uniform(0, 1) < prTerm:
					next = nodes.leaf()
				else: # select operation
					opname = random.choice(list(self.gen.opMap.keys()))
					next = nodes.node(opname, self.gen.opMap[opname].sname, depth=iter.depth+1)
			else:
				raise Exception("argtype %s not supported" % argtype)
			iter.args.append(next)

class generator:
	def __init__(self, yConf, mindepth, maxdepth):
		assert(mindepth < maxdepth)
		self.mindepth = mindepth
		self.depthdiff = maxdepth - mindepth
		self.opMap = {}
		for opname, sname, atypes in parseConfig(yConf):
			self.opMap[opname] = template(self, sname, atypes)

	def generate(self):
		rootOp = random.choice(list(self.opMap.keys()))
		root = nodes.node(rootOp, self.opMap[rootOp].sname)
		stack = [root]
		while len(stack) > 0:
			iter = stack.pop()
			if isinstance(iter, nodes.node):
				self.opMap[iter.name].getNexts(iter)
				stack.extend(iter.args)
		root.make_shape(randshape())
		return root

def parseConfig(yConf):
	structs = conf.getOpts(yConf)
	opMap = []
	for opClass in structs:
		for opname in opClass['names']:
			if isinstance(opClass['args'], list):
				atypes = [[argt.strip() for argt in argstr.split(",")] for argstr in opClass['args']]
			else:
				atypes = [[argt.strip() for argt in opClass['args'].split(",")]]
			opMap.append((opname, opClass['shapeclass'], atypes))
	return opMap

def allOps(yConf):
	opMap = parseConfig(yConf)
	out = {}
	for opname, sname, args in opMap:
		out[opname] = []
		for atypes in args:
			res = nodes.node(opname, sname)
			for argtype in atypes:
				scalarM = scalarPattern.search(argtype)
				if scalarM:
					dtype = scalarM.group(1)
					next = nodes.scalar(dtype)
				elif argtype == "tensor":
					next = nodes.leaf()
				else:
					raise Exception("argtype %s not supported" % argtype)
				res.args.append(next)
			out[opname].append(res)
	return out
