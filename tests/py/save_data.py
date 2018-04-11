''' Serialize data from generated tensorflow script '''

import numpy as np

import expect_pb2 as expect_pb
from proto.serial import data_pb2 as data_pb

def add_data_repo(tens, data, shape):
	tens.type = data_pb.DOUBLE
	tens.allowed_shape[:] = shape
	tens.alloced_shape[:] = shape
	arr = data_pb.double_arr()
	arr.data[:] = data
	tens.data.Pack(arr)

class profile:
	def __init__(self):
		self.pb = expect_pb.expectation_proto()

	def save(self, ntype, id, result):
		assert(isinstance(result, np.ndarray))
		assert(result.dtype == float)
		data = result.flatten()
		shape = list(result.shape)[::-1]
		if ntype == "variable":
			add_data_repo(self.pb.vars.data_map[id], data, shape)
		elif ntype == "place":
			self.pb.places[id].data[:] = data
		elif ntype == "gradient":
			add_data_repo(self.pb.grads.data_map[id], data, shape)
		elif ntype == "output":
			self.pb.result.type = data_pb.DOUBLE
			self.pb.result.allowed_shape[:] = shape
			self.pb.result.alloced_shape[:] = shape
			arr = data_pb.double_arr()
			arr.data[:] = data
			self.pb.result.data.Pack(arr)
		else:
			raise Exception("Unsupported pb message " + ntype)

	def serialize(self, fname):
		with open(fname, 'wb') as f:
			f.write(self.pb.SerializeToString())
