""" Random synthesis exectuable """

import os

from graphast.gen import generator
from tfsynth import tfsynth

MINDEPTH = os.environ['MINDEPTH'] if 'MINDEPTH' in os.environ else 1
MAXDEPTH = os.environ['MAXDEPTH'] if 'MAXDEPTH' in os.environ else 10

def main():
	rgen = generator("structure.yml", MINDEPTH, MAXDEPTH)
	root = rgen.generate()
	script = tfsynth(root)
	print(script)

if __name__ == "__main__":
	main()
