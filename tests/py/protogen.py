''' Generate random protobufs '''

import sys
import os
from functools import *

import numpy as np

import graphast.nodes as nodes
from graphast.gen import generator, allOps
from save_ast import save_ast
from save_data import add_data_repo
from utils import traverse

from proto.serial import data_pb2 as data_pb
from proto.serial import graph_pb2 as graph_pb

MINDEPTH = os.environ['MINDEPTH'] if 'MINDEPTH' in os.environ else 1 
MAXDEPTH = os.environ['MAXDEPTH'] if 'MAXDEPTH' in os.environ else 10
GRAPH_EXT = ".graph"
REGISTRY = "registry.txt"

def save_data(graphpb, outpath):
	pb = data_pb.data_repo_proto()
	for id in graphpb.node_map:
		nodepb = graphpb.node_map[id]
		if nodepb.type == graph_pb.node_proto.VARIABLE:
			varpb = graph_pb.variable_proto()
			nodepb.detail.Unpack(varpb)
			shape = varpb.allowed_shape
			assert(len(shape))
			nelems = reduce(lambda x, y: x * y, shape)
			data = np.random.rand(nelems)
			add_data_repo(pb.data_map[id], data, shape)
	with open(os.path.join(outpath, "RANDOM.data"), 'wb') as f:
		f.write(pb.SerializeToString())

def main():
	if len(sys.argv) < 2:
		print("Usage:", sys.argv[0], "<output/path>")
		sys.exit(-1)
	outpath = sys.argv[1]

	ops = allOps("structure.yml")
	rgen = generator("structure.yml", MINDEPTH, MAXDEPTH) 
	root = rgen.generate()
	ops["RANDOM"] = [root]

	if not os.path.isdir(outpath):
		os.makedirs(outpath)

	with open(os.path.join(outpath, REGISTRY), 'w') as reg:
		for opname in ops: 
			tops = ops[opname]
			for i in range(len(tops)):
				fname = opname
				if i > 0:
					fname = fname + str(i)
				fname = fname + GRAPH_EXT
				root = tops[i]
				with open(os.path.join(outpath, fname), 'wb') as f:
					graphpb = save_ast(root)
					f.write(graphpb.SerializeToString())
					if opname == "RANDOM":
						save_data(graphpb, outpath)
				reg.write(fname + '\n')
		
if __name__ == "__main__":
	main()
