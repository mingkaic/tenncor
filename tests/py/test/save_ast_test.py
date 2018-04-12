''' Test graphast save_ast utility '''

import unittest

from proto.serial import graph_pb2 as graph_pb

from graphast.nodes import leaf
from graphast.gen import allOps
from tenncorgen.save_ast import save_ast

class TestSaveAst(unittest.TestCase):
	def test_save_ast(self):
		ops = allOps("structure.yml")
		for opname in ops:
			tops = ops[opname]
			for root in tops:
				graphpb = save_ast(root)
				vars = graphpb.create_order[:-1]
				for var in vars:
					self.assertTrue(var in graphpb.node_map)
					nodepb = graphpb.node_map[var]
					self.assertNotEqual(graph_pb.NodePb.FUNCTOR, nodepb.type)
				rootpb = graphpb.node_map[graphpb.create_order[-1]]
				self.assertEqual(graph_pb.NodePb.FUNCTOR, rootpb.type)
				funcpb = graph_pb.FunctorPb()
				rootpb.detail.Unpack(funcpb)
				self.assertEqual(root.name, graph_pb.OpcodeT.Name(funcpb.opcode))
