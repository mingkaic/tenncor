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
	std::vector<size_t> fragranks = get_int(3, "fragranks", {0, 4});
	std::vector<size_t> outer1 = get_int(fragranks[0], "outer1", {1, 6});
	std::vector<size_t> inner = get_int(fragranks[1], "inner", {1, 6});
	std::vector<size_t> outer2 = get_int(fragranks[2], "outer2", {1, 6});
	std::vector<size_t> clist = outer1;
	clist.insert(clist.end(), inner.begin(), inner.end());
	clist.insert(clist.end(), outer2.begin(), outer2.end());

	clay::Shape shape(clist);
	mold::Range range(fragranks[0], fragranks[0] + fragranks[1]);
	clay::Shape expect(inner);
	clay::Shape inners = range.apply(shape);
	EXPECT_SHAPEQ(expect, inners);

	mold::Range halfin(fragranks[0], clist.size());
	std::vector<size_t> halfvec = outer1;
	halfvec.insert(halfvec.end(), inner.begin(), inner.end());
	clay::Shape halfinner = halfin.apply(clay::Shape(halfvec));
	EXPECT_SHAPEQ(inners, halfinner);

	mold::Range allout(fragranks[0] + fragranks[1], clist.size());
	clay::Shape outinner = allout.apply(clay::Shape(outer1));
	EXPECT_FALSE(outinner.is_part_defined()) <<
		"out of range inner shape is defined: " << clay::to_string(outinner);
}


TEST_F(RANGE, Front_)
{
	clay::Shape scalar = get_int(1, "scalar", {1, 6});
	mold::Range srange(0, 0);
	clay::Shape undef = srange.front(scalar);
	EXPECT_FALSE(undef.is_part_defined()) <<
		"scalar shape with scalar outer range is defined: " << clay::to_string(undef);

	std::vector<size_t> fragranks = get_int(3, "fragranks", {0, 4});
	std::vector<size_t> outer1 = get_int(fragranks[0], "outer1", {1, 6});
	std::vector<size_t> inner = get_int(fragranks[1], "inner", {1, 6});
	std::vector<size_t> outer2 = get_int(fragranks[2], "outer2", {1, 6});
	std::vector<size_t> clist = outer1;
	clist.insert(clist.end(), inner.begin(), inner.end());
	clist.insert(clist.end(), outer2.begin(), outer2.end());

	clay::Shape shape(clist);
	mold::Range range(fragranks[0], fragranks[0] + fragranks[1]);
	clay::Shape expect(outer1);
	clay::Shape ot1 = range.front(shape);
	EXPECT_SHAPEQ(expect, ot1);

	mold::Range halfin(fragranks[0], clist.size());
	std::vector<size_t> halfvec = outer1;
	halfvec.insert(halfvec.end(), inner.begin(), inner.end());
	clay::Shape halfinner = halfin.front(clay::Shape(halfvec));
	EXPECT_SHAPEQ(expect, halfinner);

	mold::Range allout(fragranks[0] + fragranks[1], clist.size());
	clay::Shape outinner = allout.front(clay::Shape(outer1));
	EXPECT_SHAPEQ(expect, outinner);
}


TEST_F(RANGE, Back_)
{
	std::vector<size_t> fragranks = get_int(3, "fragranks", {0, 4});
	std::vector<size_t> outer1 = get_int(fragranks[0], "outer1", {1, 6});
	std::vector<size_t> inner = get_int(fragranks[1], "inner", {1, 6});
	std::vector<size_t> outer2 = get_int(fragranks[2], "outer2", {1, 6});
	std::vector<size_t> clist = outer1;
	clist.insert(clist.end(), inner.begin(), inner.end());
	clist.insert(clist.end(), outer2.begin(), outer2.end());

	clay::Shape shape(clist);
	mold::Range range(fragranks[0], fragranks[0] + fragranks[1]);
	clay::Shape expect(outer2);
	clay::Shape ot2 = range.back(shape);
	EXPECT_SHAPEQ(expect, ot2);

	mold::Range halfin(fragranks[0], clist.size());
	std::vector<size_t> halfvec = outer1;
	halfvec.insert(halfvec.end(), inner.begin(), inner.end());
	clay::Shape halfinner = halfin.back(clay::Shape(halfvec));
	EXPECT_FALSE(halfinner.is_part_defined()) <<
		"partial range second outer shape is defined: " << clay::to_string(halfinner);

	mold::Range allout(fragranks[0] + fragranks[1], clist.size());
	clay::Shape outinner = allout.back(clay::Shape(outer1));
	EXPECT_FALSE(outinner.is_part_defined()) <<
		"out of range second outer shape is defined: " << clay::to_string(outinner);
}


#endif /* DISABLE_RANGE_TEST */


#endif /* DISABLE_MOLD_MODULE_TESTS */
