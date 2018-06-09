#ifndef DISABLE_CLAY_MODULE_TESTS

#include "gtest/gtest.h"

#include "testify/mocker/mocker.hpp"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/tensor.hpp"
#include "clay/memory.hpp"
#include "clay/error.hpp"


#ifndef DISABLE_CLAY_TEST


using namespace testutil;


class TENSOR : public fuzz_test
{
protected:
	virtual void SetUp (void) {}

	virtual void TearDown (void)
	{
		testutil::fuzz_test::TearDown();
		testify::mocker::clear();
	}
};


// cover Tensor: constructor, moves, get_shape, get_type, get_state
TEST_F(TENSOR, Constructor_C000)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	std::vector<size_t> clist = random_def_shape(this, {1, 7});
	std::vector<size_t> plist = make_partial(this, clist);

	clay::Shape undef;
	clay::Shape pshape(plist);
	clay::Shape cshape(clist);

	size_t nbytes = cshape.n_elems() * clay::type_size(dtype);
	std::string s1 = get_string(nbytes, "s1");
	clay::Tensor ten(cshape, dtype);
	clay::State state = ten.get_state();
	memcpy(state.data_.lock().get(),
		s1.c_str(), nbytes);

	EXPECT_THROW(clay::Tensor(undef, dtype), clay::InvalidShapeError);
	EXPECT_THROW(clay::Tensor(plist, dtype), clay::InvalidShapeError);
	EXPECT_THROW(clay::Tensor(cshape, clay::DTYPE::BAD), clay::UnsupportedTypeError);

	clay::Shape gotshape = ten.get_shape();
	clay::DTYPE gottype = ten.get_type();
	EXPECT_SHAPEQ(cshape,  gotshape);
	EXPECT_EQ(dtype, gottype);

	EXPECT_SHAPEQ(cshape,  state.shape_);
	EXPECT_EQ(dtype, state.dtype_);

	clay::DTYPE dtype2 = (clay::DTYPE) get_int(1, "dtype2", {1, clay::DTYPE::_SENTINEL - 1})[0];
	std::vector<size_t> clist2 = random_def_shape(this);
	clay::Shape cshape2(clist2);

	clay::Tensor cpassign(cshape2, dtype2);
	clay::Tensor cp(ten);

	clay::State cp_state = cp.get_state();
	clay::State state2 = ten.get_state();

	EXPECT_SHAPEQ(cshape,  cp_state.shape_);
	EXPECT_EQ(dtype, cp_state.dtype_);

	cpassign = cp;

	clay::State cpa_state = cpassign.get_state();

	EXPECT_SHAPEQ(cshape,  cpa_state.shape_);
	EXPECT_EQ(dtype, cpa_state.dtype_);
}


// cover Tensor: total_bytes
TEST_F(TENSOR, TotalBytes_C001)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	std::vector<size_t> clist = random_def_shape(this, {1, 7});
	clay::Shape cshape(clist);
	size_t nbytes = cshape.n_elems() * clay::type_size(dtype);
	clay::Tensor ten(cshape, dtype);
	EXPECT_EQ(nbytes, ten.total_bytes());
}


#endif /* DISABLE_CLAY_TEST */


#endif /* DISABLE_CLAY_MODULE_TESTS */

