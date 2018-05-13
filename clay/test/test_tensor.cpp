#ifndef DISABLE_CLAY_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"
#include "fuzzutil/mock_src.hpp"

#include "clay/tensor.hpp"
#include "clay/memory.hpp"


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
	std::shared_ptr<char> data = clay::make_char(nbytes);
	memcpy(data.get(), s1.c_str(), nbytes);

	clay::Tensor ten(data, cshape, dtype);
	EXPECT_THROW(clay::Tensor(nullptr, cshape, dtype), std::exception);
	EXPECT_THROW(clay::Tensor(data, undef, dtype), std::exception);
	EXPECT_THROW(clay::Tensor(data, plist, dtype), std::exception);
	EXPECT_THROW(clay::Tensor(data, cshape, clay::DTYPE::BAD), std::exception);

	clay::Shape gotshape = ten.get_shape();
	clay::DTYPE gottype = ten.get_type();
	EXPECT_SHAPEQ(cshape,  gotshape);
	EXPECT_EQ(dtype, gottype);

	clay::State state = ten.get_state();

	EXPECT_EQ(data, state.data_.lock());
	EXPECT_SHAPEQ(cshape,  state.shape_);
	EXPECT_EQ(dtype, state.dtype_);

	clay::DTYPE dtype2 = (clay::DTYPE) get_int(1, "dtype2", {1, clay::DTYPE::_SENTINEL - 1})[0];
	std::vector<size_t> clist2 = random_def_shape(this);
	clay::Shape cshape2(clist2);

	size_t nbytes2 = cshape2.n_elems() * clay::type_size(dtype2);
	std::shared_ptr<char> data2 = clay::make_char(nbytes2);

	clay::Tensor cpassign(data2, cshape2, dtype2);
	clay::Tensor mvassign(data2, cshape2, dtype2);
	clay::Tensor cp(ten);
	clay::Tensor mv(std::move(ten));

	clay::State cp_state = cp.get_state();
	clay::State mv_state = mv.get_state();
	clay::State state2 = ten.get_state();

	EXPECT_EQ(data, cp_state.data_.lock());
	EXPECT_SHAPEQ(cshape,  cp_state.shape_);
	EXPECT_EQ(dtype, cp_state.dtype_);

	EXPECT_EQ(data, mv_state.data_.lock());
	EXPECT_SHAPEQ(cshape,  mv_state.shape_);
	EXPECT_EQ(dtype, mv_state.dtype_);

	EXPECT_TRUE(state2.data_.expired()) << "moved ten data ptr not expired";
	EXPECT_SHAPEQ(undef,  state2.shape_);
	EXPECT_EQ(clay::DTYPE::BAD, state2.dtype_);

	cpassign = cp;
	mvassign = std::move(mv);

	clay::State cpa_state = cpassign.get_state();
	clay::State mva_state = mvassign.get_state();
	clay::State mv2 = mv.get_state();

	EXPECT_EQ(data, cpa_state.data_.lock());
	EXPECT_SHAPEQ(cshape,  cpa_state.shape_);
	EXPECT_EQ(dtype, cpa_state.dtype_);

	EXPECT_EQ(data, mva_state.data_.lock());
	EXPECT_SHAPEQ(cshape,  mva_state.shape_);
	EXPECT_EQ(dtype, mva_state.dtype_);

	EXPECT_TRUE(mv2.data_.expired()) << "moved mv data ptr not expired";
	EXPECT_SHAPEQ(undef,  mv2.shape_);
	EXPECT_EQ(clay::DTYPE::BAD, mv2.dtype_);
}


// cover Tensor: total_bytes
TEST_F(TENSOR, TotalBytes_C001)
{
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::DTYPE::_SENTINEL - 1})[0];
	std::vector<size_t> clist = random_def_shape(this, {1, 7});
	clay::Shape cshape(clist);
	size_t nbytes = cshape.n_elems() * clay::type_size(dtype);
	std::string temp = get_string(nbytes, "s1");
	std::shared_ptr<char> data = clay::make_char(nbytes);
	memcpy(data.get(), temp.c_str(), nbytes);

	clay::Tensor ten(data, cshape, dtype);
	EXPECT_EQ(nbytes, ten.total_bytes());
}


// cover tensor: read_from(const idata_src& src), write_to, get_type, has_data
TEST_F(TENSOR, ReadFrom_C002)
{
	mock_source source(this);
	std::shared_ptr<char> data = source.ptr_;
	clay::Shape shape = source.state_.shape_;
	clay::DTYPE dtype = source.state_.dtype_;
	size_t nbytes = shape.n_elems() * clay::type_size(dtype);
	std::shared_ptr<char> data2 = clay::make_char(nbytes);

	clay::Tensor ten(data2, shape, dtype);
	EXPECT_TRUE(ten.read_from(source)) << "failed to read with appropriate source";

	clay::State state = ten.get_state();
	std::string got(state.data_.lock().get(), nbytes);
	EXPECT_STREQ(source.uuid_.c_str(), got.c_str());

	mock_source baddata(nullptr, shape, dtype);
	mock_source badshape(data, clay::Shape(), dtype);
	mock_source badtype(data, shape, clay::DTYPE::BAD);

	EXPECT_FALSE(ten.read_from(baddata)) << "successful read from baddata source";
	EXPECT_FALSE(ten.read_from(badshape)) << "successful read from badshape source";
	EXPECT_FALSE(ten.read_from(badtype)) << "successful read from badtype source";

	// // assert data is unchanged after fail reads
	// clay::State state2 = ten.get_state();
	// std::string got2(state2.data_.lock().get(), nbytes);
	// EXPECT_STREQ(source.uuid_.c_str(), got2.c_str());
}


#endif /* DISABLE_CLAY_TEST */


#endif /* DISABLE_CLAY_MODULE_TESTS */

