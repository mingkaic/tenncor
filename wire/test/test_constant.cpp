#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "wire/constant.hpp"


#ifndef DISABLE_CONSTANT_TEST


using namespace testutil;


class CONSTANT : public fuzz_test {};


TEST_F(CONSTANT, GetScalar_D000)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	unsigned short bsize = clay::type_size(dtype);
	std::string data = get_string(bsize, "data");
	wire::Constant* c = nullptr;

	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
			c = wire::Constant::get(*((double*) data.c_str()));
		break;
		case clay::DTYPE::FLOAT:
			c = wire::Constant::get(*((float*) data.c_str()));
		break;
		case clay::DTYPE::INT8:
			c = wire::Constant::get(*((int8_t*) data.c_str()));
		break;
		case clay::DTYPE::UINT8:
			c = wire::Constant::get(*((uint8_t*) data.c_str()));
		break;
		case clay::DTYPE::INT16:
			c = wire::Constant::get(*((int16_t*) data.c_str()));
		break;
		case clay::DTYPE::UINT16:
			c = wire::Constant::get(*((uint16_t*) data.c_str()));
		break;
		case clay::DTYPE::INT32:
			c = wire::Constant::get(*((int32_t*) data.c_str()));
		break;
		case clay::DTYPE::UINT32:
			c = wire::Constant::get(*((uint32_t*) data.c_str()));
		break;
		case clay::DTYPE::INT64:
			c = wire::Constant::get(*((int64_t*) data.c_str()));
		break;
		case clay::DTYPE::UINT64:
			c = wire::Constant::get(*((uint64_t*) data.c_str()));
		break;
		default:
			ASSERT_FALSE(true);
	}
	ASSERT_TRUE(c->has_data());

	clay::State state = c->get_state();
	clay::Shape wun({1});
	EXPECT_SHAPEQ(wun, state.shape_);
	EXPECT_EQ(dtype, state.dtype_);
	std::string got(state.data_.lock().get(), bsize);
	EXPECT_STREQ(data.c_str(), got.c_str());

	EXPECT_THROW(wire::Constant::get(data), std::exception);

	delete c;
}


TEST_F(CONSTANT, GetVec_D001)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	clay::Shape shape = random_def_shape(this);
	size_t bytesize = clay::type_size(dtype) * shape.n_elems();
	std::string data = get_string(bytesize, "data");
	wire::Constant* v = nullptr;

	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double* ptr = (double*) data.c_str();
			std::vector<double> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float* ptr = (float*) data.c_str();
			std::vector<float> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t* ptr = (int8_t*) data.c_str();
			std::vector<int8_t> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t* ptr = (uint8_t*) data.c_str();
			std::vector<uint8_t> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t* ptr = (int16_t*) data.c_str();
			std::vector<int16_t> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t* ptr = (uint16_t*) data.c_str();
			std::vector<uint16_t> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t* ptr = (int32_t*) data.c_str();
			std::vector<int32_t> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t* ptr = (uint32_t*) data.c_str();
			std::vector<uint32_t> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t* ptr = (int64_t*) data.c_str();
			std::vector<int64_t> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t* ptr = (uint64_t*) data.c_str();
			std::vector<uint64_t> vec(ptr, ptr + shape.n_elems());
			v = wire::Constant::get(vec, shape);
		}
		break;
		default:
			ASSERT_FALSE(true);
	}
	ASSERT_TRUE(v->has_data());

	clay::State state = v->get_state();
	EXPECT_SHAPEQ(shape, state.shape_);
	EXPECT_EQ(dtype, state.dtype_);
	std::string got(state.data_.lock().get(), bytesize);
	EXPECT_STREQ(data.c_str(), got.c_str());

	std::vector<std::string> sdata(shape.n_elems(), "sample");
	EXPECT_THROW(wire::Constant::get(sdata, shape), std::exception);

	delete v;
}


#endif /* DISABLE_CONSTANT_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
