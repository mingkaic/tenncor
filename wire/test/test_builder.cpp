#ifndef DISABLE_WIRE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "wire/builder.hpp"

#include "wire/omap.hpp"


#ifndef DISABLE_BUILDER_TEST


using namespace testutil;


class BUILDER : public fuzz_test {};


TEST_F(BUILDER, Constant_B000)
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
		clay::Shape wun({1});
		EXPECT_SHAPEQ(wun, cstate.shape_);
		EXPECT_SHAPEQ(shape, vstate.shape_);
	};
	if (doub)
	{
		double scalar = get_double(1, "scalar", {-251, 120})[0];
		std::vector<double> vec = get_double(n, "vec", {-251, 120});
		c = wire::get_constant(scalar);
		v = wire::get_constant(vec, shape);
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
		c = wire::get_constant(scalar);
		v = wire::get_constant(vec, shape);
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


#endif /* DISABLE_BUILDER_TEST */


#endif /* DISABLE_WIRE_MODULE_TESTS */
