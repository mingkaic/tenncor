''' Generate random protobufs '''

import sys
import os

from graphast.gen import generator, allOps
from save_ast import save_ast

MINDEPTH = os.environ['MINDEPTH'] if 'MINDEPTH' in os.environ else 1 
MAXDEPTH = os.environ['MAXDEPTH'] if 'MAXDEPTH' in os.environ else 10
GRAPH_EXT = ".graph"
REGISTRY = "registry.txt"

def main():
	if len(sys.argv) < 2:
		print("Usage:", sys.argv[0], "<output/path>")
		sys.exit(-1)
	outpath = sys.argv[1]

	ops = allOps("structure.yml")
	rgen = generator("structure.yml", MINDEPTH, MAXDEPTH) 
	root = rgen.generate()
	ops["random"] = [root]

	if not os.path.isdir(outpath):
		os.makedirs(outpath)

	with open(os.path.join(outpath, REGISTRY), 'w') as reg:
		for opname in ops: 
			tops = ops[opname]
			for i in range(len(tops)):
				fname = opname.lower()
				if i > 0:
					fname = fname + str(i)
				fname = fname + GRAPH_EXT
				with open(os.path.join(outpath, fname), 'wb') as f:
					graphpb = save_ast(root)
					f.write(graphpb.SerializeToString())
				reg.write(fname + '\n')
		

if __name__ == "__main__":
	main()
