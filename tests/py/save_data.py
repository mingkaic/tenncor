''' Serialize data from generated tensorflow script '''

from proto.serial import data_pb2 as data_pb

class profile:
	def __init__(self):
		self.pb = data_pb.data_repo_proto()

	def save(ntype, id, result):
		if ntype == "variable":
			# self.pb[id] = 
			pass
		elif ntype == "scalar":
			pass
		elif ntype == "gradient":
			pass
		elif ntype == "output":
			pass
		else:
			raise Exception("Unsupported pb message " + ntype)

	def serialize(self, fname):
		with open(fname, 'wb') as f:
			f.write(self.pb.SerializeToString())
