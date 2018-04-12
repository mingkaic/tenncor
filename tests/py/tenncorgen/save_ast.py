''' Save generated graphast to protobuf '''

import os
import random
import string

from proto.serial import data_pb2 as data_pb
from proto.serial import graph_pb2 as graph_pb

import graphast.nodes as nodes
from tenncorgen.utils import traverse

MINDEPTH = os.environ['MINDEPTH'] if 'MINDEPTH' in os.environ else 1
MAXDEPTH = os.environ['MAXDEPTH'] if 'MAXDEPTH' in os.environ else 10

def _randVariable(n):
	postfix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n-1))
	return random.choice(string.ascii_uppercase + string.ascii_lowercase) + postfix

def _func_process(nodepb, func, deps):
	funcpb = graph_pb.FunctorPb()
	funcpb.opcode = graph_pb.OpcodeT.Value(func.name)
	funcpb.args[:] = deps
	nodepb.detail.Pack(funcpb)

def _var_process(nodepb, varpos, var):
	varpb = graph_pb.VariablePb()
	varpb.varpos = varpos
	varpb.source.src = graph_pb.SourcePb.UNIFORM
	varpb.source.dtype = data_pb.DOUBLE
	arr = data_pb.DoubleArr()
	arr.data[:] = [-1.0, 1.0]
	varpb.source.settings.Pack(arr)
	varpb.allowed_shape[:] = var.shape
	nodepb.detail.Pack(varpb)

def _const_process(nodepb, value):
	tensorpb = data_pb.TensorPb()
	arr = data_pb.Int32Arr()
	arr.data[:] = [value]
	tensorpb.data.Pack(arr)
	tensorpb.type = data_pb.INT32
	tensorpb.allowed_shape[:] = [1]
	tensorpb.alloced_shape[:] = [1]
	nodepb.detail.Pack(tensorpb)

def _place_process(nodepb):
	placepb = graph_pb.PlacePb()
	placepb.allowed_shape[:] = [1]
	nodepb.detail.Pack(placepb)

def save_ast(root):
	gid = _randVariable(16)
	graphpb = graph_pb.GraphPb()
	graphpb.gid = gid
	def proc_node(node, deps):
		id = _randVariable(16)
		graphpb.create_order.append(id)
		nodepb = graphpb.node_map[id]
		if len(deps):
			nodepb.type = graph_pb.NodePb.FUNCTOR
			nodepb.label = node.name
			_func_process(nodepb, node, deps)
		elif isinstance(node, nodes.leaf):
			nodepb.type = graph_pb.NodePb.VARIABLE
			nodepb.label = 'variable'
			_var_process(nodepb, id, node)
		elif isinstance(node, nodes.scalar):
			try:
				value = int(node.value)
				nodepb.type = graph_pb.NodePb.CONSTANT
				nodepb.label = 'scalar'
				_const_process(nodepb, value)
			except ValueError:
				nodepb.type = graph_pb.NodePb.PLACEHOLDER
				nodepb.label = 'scalar'
				_place_process(nodepb)
		else:
			raise Exception("supported node type")
		return id, None

	traverse(root, proc_node)
	return graphpb
