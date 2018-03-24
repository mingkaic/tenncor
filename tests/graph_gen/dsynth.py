""" Deterministic synthesis exectuable """

from gen import allOps
import synth

def main():
	ops = allOps("structure.yml")
	for opname in ops:
		tops = ops[opname]
		for res in tops:
			synth.synth(res)

if __name__ == "__main__":
	main()
