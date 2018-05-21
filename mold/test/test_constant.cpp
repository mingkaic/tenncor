#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/dtype.hpp"
#include "mold/constant.hpp"


#ifndef DISABLE_CONSTANT_TEST


using namespace testutil;


class CONSTANT : public fuzz_test {};


TEST_F(CONSTANT, Bad_B000)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape = clist;
	clay::Shape part = make_partial(this, clist);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	size_t nbytes = clay::type_size(dtype) * shape.n_elems();
	std::shared_ptr<char> ptr = clay::make_char(nbytes);
	
	EXPECT_THROW(mold::Constant(nullptr, shape, dtype), std::exception);
	EXPECT_THROW(mold::Constant(ptr, part, dtype), std::exception);
	EXPECT_THROW(mold::Constant(ptr, shape, clay::DTYPE::BAD), std::exception);
}


TEST_F(CONSTANT, Copy_B001)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	clay::Shape shape = random_def_shape(this);
	size_t nbytes = clay::type_size(dtype) * shape.n_elems();
	std::string data = get_string(nbytes, "data");
	std::shared_ptr<char> ptr = clay::make_char(nbytes);
	memcpy(ptr.get(), data.c_str(), nbytes);

	mold::Constant con(ptr, shape, dtype);
	void* origdata = (void*)ptr.get();

	mold::Constant cp(con);
	clay::State cstate = cp.get_state();
	EXPECT_SHAPEQ(shape, cstate.shape_);
	EXPECT_EQ(dtype, cstate.dtype_);
	const char* gotdata = cstate.data_.lock().get();
	EXPECT_NE(origdata, (void*) gotdata);
	std::string cgot(gotdata, nbytes);
	EXPECT_STREQ(data.c_str(), cgot.c_str());
}


TEST_F(CONSTANT, Data_B002)
{
	bool doub = get_int(1, "doub", {0, 1})[0];
	mold::iNode* node;
	clay::Shape shape = random_def_shape(this);
	size_t n = shape.n_elems();
	auto interm_check = [&]()
	{
		ASSERT_TRUE(node->has_data());
		clay::State state = node->get_state();
		EXPECT_SHAPEQ(shape, state.shape_);
	};
	if (doub)
	{
		size_t nbytes = n * sizeof(double);
		std::vector<double> data = get_double(n, "data", {-251, 120});
		std::shared_ptr<char> ptr = clay::make_char(nbytes);
		memcpy(ptr.get(), &data[0], nbytes);
		node = new mold::Constant(ptr, shape, clay::DTYPE::DOUBLE);
		interm_check();
	
		clay::State state = node->get_state();
		EXPECT_EQ(clay::DTYPE::DOUBLE, state.dtype_);
		double* got = (double*) state.data_.lock().get();
		std::vector<double> gvec(got, got + n);
		EXPECT_ARREQ(data, gvec);
	}
	else
	{
		size_t nbytes = n * sizeof(uint64_t);
		std::vector<size_t> temp = get_int(n, "data", {0, 1220});
		std::vector<uint64_t> data(temp.begin(), temp.end());
		std::shared_ptr<char> ptr = clay::make_char(nbytes);
		memcpy(ptr.get(), &data[0], nbytes);
		node = new mold::Constant(ptr, shape, clay::DTYPE::UINT64);
		interm_check();
	
		clay::State state = node->get_state();
		EXPECT_EQ(clay::DTYPE::UINT64, state.dtype_);
		uint64_t* got = (uint64_t*) state.data_.lock().get();
		std::vector<uint64_t> gvec(got, got + n);
		EXPECT_ARREQ(data, gvec);
	}

	delete node;
}


TEST_F(CONSTANT, Derive_B003)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	unsigned short bsize = clay::type_size(dtype);
	std::shared_ptr<char> ptr = clay::make_char(bsize);
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t d = 10;
			std::memcpy(ptr.get(), &d, bsize);
		}
		break;
		default:
			ASSERT_TRUE(false) << "generated bad type";
		break;
	}
	mold::Constant c(ptr, clay::Shape({1}), dtype);
	mold::iNode* zaro = c.derive(&c);
	ASSERT_NE(nullptr, dynamic_cast<mold::Constant*>(zaro));
	clay::State z = zaro->get_state();
	EXPECT_EQ(dtype, z.dtype_);
	std::vector<size_t> wun{1};
	EXPECT_ARREQ(wun, z.shape_.as_list());
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
		{
			double gotz = *((double*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::FLOAT:
		{
			float gotz = *((float*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::INT8:
		{
			int8_t gotz = *((int8_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::UINT8:
		{
			uint8_t gotz = *((uint8_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::INT16:
		{
			int16_t gotz = *((int16_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::UINT16:
		{
			uint16_t gotz = *((uint16_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::INT32:
		{
			int32_t gotz = *((int32_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::UINT32:
		{
			uint32_t gotz = *((uint32_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		case clay::DTYPE::INT64:
		{
			int64_t gotz = *((int64_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
			}
		break;
		case clay::DTYPE::UINT64:
		{
			uint64_t gotz = *((uint64_t*) z.data_.lock().get());
			EXPECT_EQ(0, gotz);
		}
		break;
		default:
		break;
	}
	delete zaro;
}


#endif /* DISABLE_CONSTANT_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
