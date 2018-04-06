''' Test graphast save '''

import unittest

from graphast.gen import generator
from protogen import ast_save

class TestProtogen(unittest.TestCase):
	def test_ast_save(self):
		rgen = generator("structure.yml", 1, 10)
		root = rgen.generate()
		proto = ast_save(root)
