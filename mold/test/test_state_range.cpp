#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "clay/memory.hpp"
#include "mold/state_range.hpp"


#ifndef DISABLE_STATE_RANGE_TEST


using namespace testutil;


class STATE_RANGE : public fuzz_test {};


TEST_F(STATE_RANGE, StateBehavior_)
{
	clay::Shape shape = random_def_shape(this);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::_SENTINEL-1})[0];
	std::shared_ptr<char> data = clay::make_char(shape.n_elems() * clay::type_size(dtype));
	std::vector<size_t> indices = get_int(2, "indices");
	mold::Range range(indices[0], indices[1]);
	mold::StateRange state(clay::State(data, shape, dtype), range);

	EXPECT_EQ(data.get(), state.get());
	EXPECT_SHAPEQ(shape, state.shape());
	EXPECT_EQ(dtype, state.type());
	EXPECT_EQ(range.lower_, state.drange_.lower_);
	EXPECT_EQ(range.upper_, state.drange_.upper_);
}


TEST_F(STATE_RANGE, Inner_)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape(clist);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::_SENTINEL-1})[0];
	std::shared_ptr<char> data = clay::make_char(shape.n_elems() * clay::type_size(dtype));
	size_t rank = shape.rank();
	std::vector<size_t> values = get_int(2, "rangevalues", {0, rank});
	mold::Range range(values[0], values[1]);
	mold::StateRange state(clay::State(data, shape, dtype), range);
	auto it = shape.begin();
	std::vector<size_t> exinner(it + range.lower_, it + range.upper_);
	clay::Shape expect(exinner);
	clay::Shape inner = state.inner();
	EXPECT_SHAPEQ(expect, inner);

	mold::StateRange halfstate(clay::State(data, shape, dtype), mold::Range(values[0], 2 * rank));
	std::vector<size_t> halfvec(it + values[0], shape.end());
	clay::Shape halfexpect(halfvec);
	clay::Shape halfinner = halfstate.inner();
	EXPECT_SHAPEQ(halfexpect, halfinner);

	mold::StateRange allstate(clay::State(data, shape, dtype), mold::Range(2 * rank, 4 * rank));
	clay::Shape outinner = allstate.inner();
	EXPECT_FALSE(outinner.is_part_defined()) <<
		"out of range inner shape is defined: " << clay::to_string(outinner);
}


TEST_F(STATE_RANGE, Remove_)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape(clist);
	clay::DTYPE dtype = (clay::DTYPE) get_int(1, "dtype", {1, clay::_SENTINEL-1})[0];
	std::shared_ptr<char> data = clay::make_char(shape.n_elems() * clay::type_size(dtype));
	size_t rank = shape.rank();
	std::vector<size_t> values = get_int(2, "rangevalues", {0, rank});
	mold::Range range(values[0], values[1]);
	mold::StateRange state(clay::State(data, shape, dtype), range);
	auto it = shape.begin();
	std::vector<size_t> exouter(it, it + range.lower_);
	exouter.insert(exouter.end(), it + range.upper_, shape.end());
	clay::Shape expect(exouter);
	clay::Shape outer = state.outer();
	EXPECT_SHAPEQ(expect, outer);

	mold::StateRange halfstate(clay::State(data, shape, dtype), mold::Range(values[0], 2 * rank));
	std::vector<size_t> halfvec(it, it + values[0]);
	clay::Shape halfexpect(halfvec);
	clay::Shape halfouter = halfstate.outer();
	EXPECT_SHAPEQ(halfexpect, halfouter);

	mold::StateRange allstate(clay::State(data, shape, dtype), mold::Range(2 * rank, 4 * rank));
	clay::Shape outouter = allstate.outer();
	EXPECT_SHAPEQ(shape, outouter);
}


#endif /* DISABLE_STATE_RANGE_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
