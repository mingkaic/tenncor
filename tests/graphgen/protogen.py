''' Save generated graphast to protobuf '''

import random
import string

from proto.serial import data_pb2 as data_pb
from proto.serial import graph_pb2 as graph_pb

import graphast.nodes as nodes

def _randVariable(n):
	postfix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n-1))
	return random.choice(string.ascii_uppercase + string.ascii_lowercase) + postfix

# bottom up traverse tree
def _traverse(root, declare):
	lines = []
	deps = []
	if isinstance(root, nodes.node):
		for arg in root.args:
			depid, sublines = _traverse(arg, declare)
			lines.extend(sublines)
			deps.append(depid)
	id, decl = declare(root, deps)
	lines.append(decl)
	return id, lines

def _func_process(nodepb, func, deps):
    pass

def _var_process(nodepb, varpos, var):
    varpb = graph_pb.variable_proto()
    varpb.varpos = varpos
    varpb.source.src = graph_pb.source_proto.UNIFORM
    varpb.source.dtype = data_pb.FLOAT
    arr = data_pb.float_arr()
    arr.data[:] = [-1.0, 1.0]
    varpb.source.settings.Pack(arr)
    varpb.allowed_shape[:] = var.shape
    nodepb.detail.Pack(varpb)

def _const_process(nodepb, value):
    tensorpb = data_pb.tensor_proto()
    arr = data_pb.int32_arr()
    arr.data[:] = [value]
    tensorpb.data.Pack(arr)
    tensorpb.type = data_pb.INT32
    tensorpb.allowed_shape[:] = [1]
    tensorpb.alloced_shape[:] = [1]
    nodepb.detail.Pack(tensorpb)

def _place_process(nodepb):
    placepb = graph_pb.place_proto()
    placepb.allowed_shape[:] = [1]
    nodepb.detail.Pack(placepb)

def ast_save(root):
    gid = _randVariable(16)
    graphpb = graph_pb.graph_proto()
    graphpb.gid = gid
    def declare(node, deps):
        id = _randVariable(16)
        graphpb.create_order.append(id)
        nodepb = graphpb.node_map[id]
        varpos = None
        if len(deps):
            nodepb.type = graph_pb.node_proto.FUNCTOR
            nodepb.label = node.name
            _func_process(nodepb, node, deps)
        elif isinstance(node, nodes.leaf):
            nodepb.type = graph_pb.node_proto.VARIABLE
            nodepb.label = 'variable'
            varpos = _randVariable(16)
            _var_process(nodepb, varpos, node)
        elif isinstance(node, nodes.scalar):
            try:
                value = int(node.value)
                nodepb.type = graph_pb.node_proto.CONSTANT
                nodepb.label = 'scalar'
                _const_process(nodepb, value)
            except ValueError:
                nodepb.type = graph_pb.node_proto.PLACEHOLDER
                nodepb.label = 'scalar'
                _place_process(nodepb)
        else:
            raise Exception("supportede node type")
        return id, varpos

    _traverse(root, declare)
    return graphpb
