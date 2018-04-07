''' Generate random protobufs '''

import sys
import os

from graphast.gen import generator
from save_ast import save_ast

MINDEPTH = os.environ['MINDEPTH'] if 'MINDEPTH' in os.environ else 1 
MAXDEPTH = os.environ['MAXDEPTH'] if 'MAXDEPTH' in os.environ else 10 
 
def main():
	if len(sys.argv) < 2:
		print("Usage:", sys.argv[0], "PROTOBUF FILENAME")
		sys.exit(-1)

	rgen = generator("structure.yml", MINDEPTH, MAXDEPTH) 
	root = rgen.generate()
	graphpb = save_ast(root)

	with open(sys.argv[1], 'wb') as f:
		f.write(graphpb.SerializeToString())

if __name__ == "__main__":
	main()
