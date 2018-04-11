''' Test save data '''

import numpy as np
import unittest

from save_data import profile

from proto.serial import data_pb2 as data_pb

class TestSaveAst(unittest.TestCase):
	def test_save_ast(self):
		prof = profile()
		vardata = np.random.uniform(low=-1.0, high=1.0, size=(2, 3))
		gdata = np.random.uniform(low=-1.0, high=1.0, size=(2, 3))
		pdata = np.random.uniform(low=-1.0, high=1.0, size=(2, 3))
		out = np.random.uniform(low=-1.0, high=1.0, size=(2, 3))
		prof.save("variable", "samplevar", vardata)
		prof.save("gradient", "samplegrad", gdata)
		prof.save("place", "sampleplace", pdata)
		prof.save("output", "sampleout", out)
		self.assertTrue("samplevar" in prof.pb.vars.data_map)
		self.assertTrue("samplegrad" in prof.pb.grads.data_map)
		self.assertTrue("sampleplace" in prof.pb.places)
	
		vpb = prof.pb.vars.data_map["samplevar"]
		gpb = prof.pb.grads.data_map["samplegrad"]
		parr = prof.pb.places["sampleplace"]
		rpb = prof.pb.result
		self.assertTrue(np.array_equal(vpb.allowed_shape, [3, 2]))
		self.assertTrue(np.array_equal(vpb.alloced_shape, [3, 2]))
		self.assertTrue(np.array_equal(gpb.allowed_shape, [3, 2]))
		self.assertTrue(np.array_equal(gpb.alloced_shape, [3, 2]))
		self.assertTrue(np.array_equal(rpb.allowed_shape, [3, 2]))
		self.assertTrue(np.array_equal(rpb.alloced_shape, [3, 2]))
		self.assertEqual(data_pb.DOUBLE, vpb.type)
		self.assertEqual(data_pb.DOUBLE, gpb.type)
		self.assertEqual(data_pb.DOUBLE, rpb.type)

		varr = data_pb.double_arr()
		garr = data_pb.double_arr()
		rarr = data_pb.double_arr()
		vpb.data.Unpack(varr)
		gpb.data.Unpack(garr)
		rpb.data.Unpack(rarr)
		self.assertTrue(np.array_equal(varr.data, vardata.flatten()))
		self.assertTrue(np.array_equal(garr.data, gdata.flatten()))
		self.assertTrue(np.array_equal(parr.data, pdata.flatten()))
		self.assertTrue(np.array_equal(rarr.data, out.flatten()))
