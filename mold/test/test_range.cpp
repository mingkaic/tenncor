#ifndef DISABLE_MOLD_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/sgen.hpp"
#include "fuzzutil/check.hpp"

#include "mold/range.hpp"


#ifndef DISABLE_RANGE_TEST


using namespace testutil;


class RANGE : public fuzz_test {};


TEST_F(RANGE, Constructor_G000)
{
	std::vector<size_t> indices = get_int(2, "indices");
	size_t mmin = *(std::min_element(indices.begin(), indices.end()));
	size_t mmax = *(std::max_element(indices.begin(), indices.end()));
	mold::Range proper(mmin, mmax);
	mold::Range improper(mmax, mmin);

	EXPECT_EQ(proper.lower_, improper.lower_);
	EXPECT_EQ(proper.upper_, improper.upper_);
}


TEST_F(RANGE, Apply_G001)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape(clist);
	size_t rank = shape.rank();
	std::vector<size_t> values = get_int(2, "rangevalues", {0, rank});
	mold::Range range(values[0], values[1]);
	auto it = shape.begin();
	std::vector<size_t> exinner(it + range.lower_, it + range.upper_);
	clay::Shape expect(exinner);
	clay::Shape inner = range.apply(shape);
	EXPECT_SHAPEQ(expect, inner);

	mold::Range halfin(values[0], 2 * rank);
	std::vector<size_t> halfvec(it + values[0], shape.end());
	clay::Shape halfexpect(halfvec);
	clay::Shape halfinner = halfin.apply(shape);
	EXPECT_SHAPEQ(halfexpect, halfinner);

	mold::Range allout(2 * rank, 4 * rank);
	clay::Shape outinner = allout.apply(shape);
	EXPECT_FALSE(outinner.is_part_defined()) <<
		"out of range inner shape is defined: " << clay::to_string(outinner);
}


TEST_F(RANGE, Remove_G002)
{
	std::vector<size_t> clist = random_def_shape(this);
	clay::Shape shape(clist);
	size_t rank = shape.rank();
	std::vector<size_t> values = get_int(2, "rangevalues", {0, rank});
	mold::Range range(values[0], values[1]);
	auto it = shape.begin();
	std::vector<size_t> exouter(it, it + range.lower_);
	exouter.insert(exouter.end(), it + range.upper_, shape.end());
	clay::Shape expect(exouter);
	clay::Shape outer = range.remove(shape);
	EXPECT_SHAPEQ(expect, outer);

	mold::Range halfin(values[0], 2 * rank);
	std::vector<size_t> halfvec(it, it + values[0]);
	clay::Shape halfexpect(halfvec);
	clay::Shape halfouter = halfin.remove(shape);
	EXPECT_SHAPEQ(halfexpect, halfouter);

	mold::Range allout(2 * rank, 4 * rank);
	clay::Shape outouter = allout.remove(shape);
	EXPECT_SHAPEQ(shape, outouter);
}


#endif /* DISABLE_RANGE_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
