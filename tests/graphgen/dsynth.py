""" Deterministic synthesis exectuable """

from graphast.gen import allOps
from tfsynth import tfsynth

def main():
	ops = allOps("structure.yml")
	for opname in ops:
		tops = ops[opname]
		for res in tops:
			script = tfsynth(res)
			print(script)

if __name__ == "__main__":
	main()
