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


TEST_F(CONSTANT, Data_B000)
{
	bool doub = get_int(1, "doub", {0, 1})[0];
	mold::iNode* c;
	mold::iNode* v;
	clay::Shape shape = random_def_shape(this);
	size_t n = shape.n_elems();
	auto interm_check = [&]()
	{
		ASSERT_TRUE(c->has_data());
		ASSERT_TRUE(v->has_data());

		clay::State cstate = c->get_state();
		clay::State vstate = v->get_state();
		std::vector<size_t> wun{1};
		EXPECT_ARREQ(wun, cstate.shape_.as_list());
		EXPECT_SHAPEQ(shape, vstate.shape_);
	};
	if (doub)
	{
		double scalar = get_double(1, "scalar", {-251, 120})[0];
		std::vector<double> vec = get_double(n, "vec", {-251, 120});
		c = mold::make_constant(scalar);
		v = mold::make_constant(vec, shape);
		interm_check();
	
		clay::State cstate = c->get_state();
		clay::State vstate = v->get_state();
		EXPECT_EQ(clay::DTYPE::DOUBLE, cstate.dtype_);
		EXPECT_EQ(clay::DTYPE::DOUBLE, vstate.dtype_);
		double got = *((double*) cstate.data_.lock().get());
		double* gotv = (double*) vstate.data_.lock().get();
		EXPECT_EQ(scalar, got);
		std::vector<double> gvec(gotv, gotv + n);
		EXPECT_ARREQ(vec, gvec);
	}
	else
	{
		uint64_t scalar = get_int(1, "scalar", {0, 1220})[0];
		std::vector<size_t> temp = get_int(n, "vec", {0, 1220});
		std::vector<uint64_t> vec(temp.begin(), temp.end());
		c = mold::make_constant(scalar);
		v = mold::make_constant(vec, shape);
		interm_check();
	
		clay::State cstate = c->get_state();
		clay::State vstate = v->get_state();
		EXPECT_EQ(clay::DTYPE::UINT64, cstate.dtype_);
		EXPECT_EQ(clay::DTYPE::UINT64, vstate.dtype_);
		uint64_t got = *((uint64_t*) cstate.data_.lock().get());
		uint64_t* gotv = (uint64_t*) vstate.data_.lock().get();
		EXPECT_EQ(scalar, got);
		std::vector<double> gvec(gotv, gotv + n);
		EXPECT_ARREQ(vec, gvec);
	}

	delete c;
	delete v;
}


TEST_F(CONSTANT, Derive_B001)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	mold::iNode* c;
	switch (dtype)
	{
		case clay::DTYPE::DOUBLE:
			c = mold::make_constant((double) 10);
		break;
		case clay::DTYPE::FLOAT:
			c = mold::make_constant((float) 10);
		break;
		case clay::DTYPE::INT8:
			c = mold::make_constant((int8_t) 10);
		break;
		case clay::DTYPE::UINT8:
			c = mold::make_constant((uint8_t) 10);
		break;
		case clay::DTYPE::INT16:
			c = mold::make_constant((int16_t) 10);
		break;
		case clay::DTYPE::UINT16:
			c = mold::make_constant((uint16_t) 10);
		break;
		case clay::DTYPE::INT32:
			c = mold::make_constant((int32_t) 10);
		break;
		case clay::DTYPE::UINT32:
			c = mold::make_constant((uint32_t) 10);
		break;
		case clay::DTYPE::INT64:
			c = mold::make_constant((int64_t) 10);
		break;
		case clay::DTYPE::UINT64:
			c = mold::make_constant((uint64_t) 10);
		break;
		default:
			c = nullptr;
			ASSERT_TRUE(false) << "generated bad type";
		break;
	}
	mold::iNode* zaro = c->derive(c);
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
}


#endif /* DISABLE_CONSTANT_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
