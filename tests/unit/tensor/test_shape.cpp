//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_TENSOR_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"


#ifndef DISABLE_SHAPE_TEST


class TENSORSHAPE : public testutils::fuzz_test {};


using namespace testutils;


// cover tensorshape:
// default and vector constructor
// clone, and vector assignment
TEST_F(TENSORSHAPE, Copy_A000)
{
	nnet::tensorshape incom_assign;
	nnet::tensorshape pcom_assign;
	nnet::tensorshape com_assign;

	nnet::tensorshape incom_vassign;
	nnet::tensorshape pcom_vassign;
	nnet::tensorshape com_vassign;

	// define shapes
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	incom_vassign = std::vector<size_t>{};
	pcom_vassign = pds;
	com_vassign = cds;

	nnet::tensorshape incom_cpy(incom_ts);
	nnet::tensorshape pcom_cpy(pcom_ts);
	nnet::tensorshape com_cpy(com_ts);

	incom_assign = incom_ts;
	pcom_assign = pcom_ts;
	com_assign = com_ts;

	EXPECT_TRUE(tensorshape_equal(incom_cpy, incom_ts)) <<
		testutils::sprintf("expecting shape %p, got shape %p", &incom_cpy, &incom_ts);
	EXPECT_TRUE(tensorshape_equal(pcom_cpy, pcom_ts)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_cpy, &pcom_ts);
	EXPECT_TRUE(tensorshape_equal(com_cpy, com_ts)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_cpy, &com_ts);
	EXPECT_TRUE(tensorshape_equal(incom_assign, incom_ts)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &incom_assign, &incom_ts);
	EXPECT_TRUE(tensorshape_equal(pcom_assign, pcom_ts)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_assign, &pcom_ts);
	EXPECT_TRUE(tensorshape_equal(com_assign, com_ts)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_assign, &com_ts);
	EXPECT_TRUE(tensorshape_equal(incom_vassign, incom_ts)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &incom_vassign, &incom_ts);
	EXPECT_TRUE(tensorshape_equal(pcom_vassign, pcom_ts)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_vassign, &pcom_ts);
	EXPECT_TRUE(tensorshape_equal(com_vassign, com_ts)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_vassign, &com_ts);

	EXPECT_TRUE(tensorshape_equal(pcom_cpy, pds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_cpy, &pds);
	EXPECT_TRUE(tensorshape_equal(com_cpy, cds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_cpy, &cds);
	EXPECT_TRUE(tensorshape_equal(pcom_assign, pds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_assign, &pds);
	EXPECT_TRUE(tensorshape_equal(com_assign, cds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_assign, &cds);
	EXPECT_TRUE(tensorshape_equal(pcom_vassign, pds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_vassign, &pds);
	ASSERT_TRUE(tensorshape_equal(com_vassign, cds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_vassign, &cds);
}


// cover tensorshape:
// default and vector constructor, and move
TEST_F(TENSORSHAPE, Move_A000)
{
	nnet::tensorshape pcom_assign;
	nnet::tensorshape com_assign;
	std::vector<size_t> empty;
	// define shapes
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	nnet::tensorshape pcom_mv(std::move(pcom_ts));
	nnet::tensorshape com_mv(std::move(com_ts));

	EXPECT_TRUE(tensorshape_equal(pcom_mv, pds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_mv, &pds);
	EXPECT_TRUE(tensorshape_equal(com_mv, cds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_mv, &cds);
	EXPECT_TRUE(tensorshape_equal(pcom_ts, empty)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_ts, &empty);
	EXPECT_TRUE(tensorshape_equal(com_ts, empty)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_ts, &empty);

	pcom_assign = std::move(pcom_mv);
	com_assign = std::move(com_mv);

	EXPECT_TRUE(tensorshape_equal(pcom_assign, pds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_assign, &pds);
	EXPECT_TRUE(tensorshape_equal(com_assign, cds)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_assign, &cds);
	EXPECT_TRUE(tensorshape_equal(pcom_mv, empty)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &pcom_mv, &empty);
	ASSERT_TRUE(tensorshape_equal(com_mv, empty)) << 
		testutils::sprintf("expecting shape %p, got shape %p", &com_mv, &empty);
}


// covers tensorshape: operator []
TEST_F(TENSORSHAPE, IndexAccessor_A001)
{
	std::vector<size_t> svec = random_def_shape(this);
	nnet::tensorshape shape(svec);
	size_t dim = get_int(1, "dim", {0, shape.rank()-1})[0];
	EXPECT_EQ(svec[dim], shape[dim]);
	EXPECT_THROW(shape[shape.rank()], std::out_of_range);
}


// covers tensorshape: as_list
TEST_F(TENSORSHAPE, AsList_A002)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	std::vector<size_t> ires = incom_ts.as_list();
	std::vector<size_t> pres = pcom_ts.as_list();
	std::vector<size_t> cres = com_ts.as_list();

	ASSERT_TRUE(ires.empty()) << 
		testutils::sprintf("expecting empty, got %vd", &ires);
	ASSERT_TRUE(std::equal(pds.begin(), pds.end(), pres.begin())) << 
		testutils::sprintf("expecting %vd, got %vd", &pds, &pres);
	ASSERT_TRUE(std::equal(cds.begin(), cds.end(), cres.begin())) << 
		testutils::sprintf("expecting %vd, got %vd", &cds, &cres);
}


// covers tensorshape: n_elems
TEST_F(TENSORSHAPE, NElems_A003)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	size_t expect_nelems = 1;
	for (size_t c : cds)
	{
		expect_nelems *= c;
	}

	size_t expect_nknown = 1;
	for (size_t p : pds)
	{
		if (p != 0)
		{
			expect_nknown *= p;
		}
	}

	EXPECT_EQ((size_t) 0, incom_ts.n_elems());
	EXPECT_EQ((size_t) 0, pcom_ts.n_elems());
	EXPECT_EQ(expect_nelems, com_ts.n_elems());

	EXPECT_EQ((size_t) 0, incom_ts.n_known());
	EXPECT_EQ(expect_nknown, pcom_ts.n_known());
	EXPECT_EQ(expect_nelems, com_ts.n_known());
}


// covers tensorshape: rank
TEST_F(TENSORSHAPE, Rank_A004)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	EXPECT_EQ((size_t) 0, incom_ts.rank());
	EXPECT_EQ(pds.size(), pcom_ts.rank());
	ASSERT_EQ(cds.size(), com_ts.rank());
}


// covers tensorshape: is_compatible_with
TEST_F(TENSORSHAPE, Compatible_A005)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	cds = random_def_shape(this, {2, 9});
	pds = make_partial(this, cds);
	cds2 = random_def_shape(this, {10, 17}, {17, std::numeric_limits<size_t>::max()});
	pds2 = make_partial(this, cds2);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);
	nnet::tensorshape pcom2_ts(pds2);
	nnet::tensorshape com2_ts(cds2);

	for (nnet::tensorshape* shape : {&incom_ts, &pcom_ts, &com_ts, &pcom2_ts, &com2_ts})
	{
		EXPECT_TRUE(incom_ts.is_compatible_with(*shape)) << 
			testutils::sprintf("expecting %p compatible with empty", &shape);
	}

	// partially defined are compatible with itself and any value in its zeros
	std::vector<size_t> cds_cpy = cds;
	std::vector<size_t> cds2_cpy = cds2;
	std::vector<size_t> cds_cpy2 = cds;
	std::vector<size_t> cds2_cpy2 = cds2;
	std::vector<size_t> brank = cds2;
	brank.push_back(cds2_cpy.back()+1);
	size_t idx1 = get_int(1, "idx1", {0, cds_cpy.size()-1})[0];
	size_t idx2 = get_int(1, "idx2", {0, cds2_cpy.size()-1})[0];
	cds_cpy[idx1] = 0;
	cds2_cpy[idx2] = 0;
	// ensure cpy2 increments are not made to indices where cpy set to 0
	// this ensure cpy2s are never compatible with cpy
	cds_cpy2[(idx1 + 1) % cds_cpy.size()]++;
	cds2_cpy2[(idx2 + 1) % cds2_cpy.size()]++;
	nnet::tensorshape fake_ps(cds_cpy);
	nnet::tensorshape fake_ps2(cds2_cpy);
	nnet::tensorshape bad_ps(cds_cpy2);
	nnet::tensorshape bad_ps2(cds2_cpy2);
	nnet::tensorshape bad_ps3(brank);

	// guarantees
	EXPECT_TRUE(fake_ps.is_compatible_with(fake_ps)) << 
		testutils::sprintf("expecting %p compatible with %p", &fake_ps, &fake_ps);
	EXPECT_TRUE(fake_ps2.is_compatible_with(fake_ps2)) << 
		testutils::sprintf("expecting %p compatible with %p", &fake_ps2, &fake_ps2);
	EXPECT_TRUE(fake_ps.is_compatible_with(com_ts)) <<
		testutils::sprintf("expecting %p compatible with %p", &com_ts, &fake_ps);
	EXPECT_TRUE(fake_ps2.is_compatible_with(com2_ts)) << 
		testutils::sprintf("expecting %p compatible with %p", &com2_ts, &fake_ps2);
	EXPECT_FALSE(fake_ps.is_compatible_with(bad_ps)) << 
		testutils::sprintf("expecting %p incompatible with %p", &bad_ps, &fake_ps);
	EXPECT_FALSE(fake_ps2.is_compatible_with(bad_ps2)) << 
		testutils::sprintf("expecting %p incompatible with %p", &bad_ps2, &fake_ps2);
	EXPECT_FALSE(fake_ps2.is_compatible_with(bad_ps3)) << 
		testutils::sprintf("expecting %p incompatible with %p", &bad_ps3, &fake_ps2);

	// fully defined are not expected to be compatible with bad
	EXPECT_TRUE(com_ts.is_compatible_with(com_ts)) << 
		testutils::sprintf("expecting %p compatible with %p", &com_ts, &com_ts);
	EXPECT_FALSE(com_ts.is_compatible_with(bad_ps)) << 
		testutils::sprintf("expecting %p incompatible with %p", &bad_ps, &com_ts);
	ASSERT_FALSE(com_ts.is_compatible_with(bad_ps2)) << 
		testutils::sprintf("expecting %p incompatible with %p", &bad_ps2, &com_ts);
}


// covers tensorshape: is_part_defined
TEST_F(TENSORSHAPE, PartDef_A006)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	random_shapes(this, pds, cds);
	random_shapes(this, pds2, cds2);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);
	nnet::tensorshape pcom2_ts(pds2);
	nnet::tensorshape com2_ts(cds2);

	ASSERT_FALSE(incom_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be undefined", &incom_ts);
	ASSERT_TRUE(pcom_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be partially defined", &pcom_ts);
	ASSERT_TRUE(pcom2_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be partially defined", &pcom2_ts);
	ASSERT_TRUE(com_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be partially defined", &com_ts);
	ASSERT_TRUE(com2_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be partially defined", &com2_ts);
}


// covers tensorshape: 
// is_fully_defined and assert_is_fully_defined
TEST_F(TENSORSHAPE, FullDef_A007)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	random_shapes(this, pds, cds);
	random_shapes(this, pds2, cds2);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape pcom2_ts(pds2);
	nnet::tensorshape com_ts(cds);
	nnet::tensorshape com2_ts(cds2);

	EXPECT_FALSE(incom_ts.is_fully_defined()) << 
		testutils::sprintf("expecting %p to not be fully defined", &incom_ts);
	EXPECT_FALSE(pcom_ts.is_fully_defined()) << 
		testutils::sprintf("expecting %p to not be fully defined", &pcom_ts);
	EXPECT_FALSE(pcom2_ts.is_fully_defined()) << 
		testutils::sprintf("expecting %p to not be fully defined", &pcom2_ts);
	EXPECT_TRUE(com_ts.is_fully_defined()) << 
		testutils::sprintf("expecting %p to be fully defined", &com_ts);
	EXPECT_TRUE(com2_ts.is_fully_defined()) << 
		testutils::sprintf("expecting %p to be fully defined", &com2_ts);

	com_ts.assert_is_fully_defined();
	com2_ts.assert_is_fully_defined();
}


// covers tensorshape: 
// assert_has_rank and assert_same_rank
TEST_F(TENSORSHAPE, RankAssert_A008)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	com_ts.assert_has_rank(cds.size());
	pcom_ts.assert_has_rank(pds.size());
	incom_ts.assert_has_rank(get_int(1, "assert random rank")[0]);

	com_ts.assert_same_rank(com_ts);
	pcom_ts.assert_same_rank(pcom_ts);
	incom_ts.assert_same_rank(com_ts);
	incom_ts.assert_same_rank(pcom_ts);
}


// covers tensorshape: 
// undefine, dependent on is_part_defined
TEST_F(TENSORSHAPE, Undefine_A009)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	EXPECT_FALSE(incom_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be undefined", &incom_ts);
	EXPECT_TRUE(pcom_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be partially defined", &pcom_ts);
	EXPECT_TRUE(com_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be partially defined", &com_ts);

	incom_ts.undefine();
	pcom_ts.undefine();
	com_ts.undefine();

	EXPECT_FALSE(incom_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be undefined", &incom_ts);
	EXPECT_FALSE(pcom_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be undefined", &pcom_ts);
	ASSERT_FALSE(com_ts.is_part_defined()) << 
		testutils::sprintf("expecting %p to be undefined", &com_ts);
}


// covers tensorshape: merge_with
TEST_F(TENSORSHAPE, Merge_A010)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	random_shapes(this, pds, cds);
	pds.push_back(1);
	random_shapes(this, pds2, cds2);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);
	nnet::tensorshape pcom2_ts(pds2);
	nnet::tensorshape com2_ts(cds2);

	// incomplete shape can merge with anything
	for (nnet::tensorshape* shape : {&pcom_ts, &com_ts, &pcom2_ts, &com2_ts, &incom_ts})
	{
		nnet::tensorshape merged = incom_ts.merge_with(*shape);
		// we're expecting merged shape to be the same as input shape
		EXPECT_TRUE(tensorshape_equal(merged, *shape)) << 
			testutils::sprintf("expecting %p, got %p", &merged, shape);
	}

	// partially defined merging with
	// fully defined yields fully defined
	// incompatible merging favors the calling member
	std::vector<size_t> cds_cpy = cds;
	std::vector<size_t> cds2_cpy = cds2;
	std::vector<size_t> cds_cpy2 = cds;
	std::vector<size_t> cds2_cpy2 = cds2;
	size_t idx1 = get_int(1, "idx1", {0, cds_cpy.size() - 1})[0];
	size_t idx2 = get_int(1, "idx2", {0, cds2_cpy.size() - 1})[0];
	cds_cpy[idx1] = 0;
	cds2_cpy[idx2] = 0;
	// ensure cpy2 increments are not made to indices where cpy set to 0
	// this ensure cpy2s are never compatible with cpy
	cds_cpy2[(idx1 + 1) % cds_cpy.size()]++;
	cds2_cpy2[(idx2 + 1) % cds2_cpy.size()]++;
	nnet::tensorshape fake_ps(cds_cpy);
	nnet::tensorshape fake_ps2(cds2_cpy);
	nnet::tensorshape incompatible(cds_cpy2);
	nnet::tensorshape incompatible2(cds2_cpy2);

	nnet::tensorshape merged = fake_ps.merge_with(com_ts);
	nnet::tensorshape merged2 = fake_ps2.merge_with(com2_ts);
	EXPECT_TRUE(tensorshape_equal(merged, com_ts)) << 
		testutils::sprintf("expecting %p, got %p", &merged, &com_ts);
	EXPECT_TRUE(tensorshape_equal(merged2, com2_ts)) << 
		testutils::sprintf("expecting %p, got %p", &merged2, &com2_ts);
	EXPECT_TRUE(tensorshape_equal(
		incompatible.merge_with(com_ts), incompatible));
	EXPECT_TRUE(tensorshape_equal(
		incompatible2.merge_with(com2_ts), incompatible2));

	// merging different ranks will error
	assert(pds.size() > cds.size()); // true by generation implementation
	EXPECT_THROW(pcom_ts.merge_with(com_ts), std::logic_error);
}


// covers tensorshape: 
// trim, dependent on rank
TEST_F(TENSORSHAPE, Trim_A011)
{
	std::vector<size_t> ids;
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	random_shapes(this, pds, cds);
	// padd a bunch of ones to pds and cds
	std::vector<size_t> npads = get_int(5, "npads", {3, 12});
	ids.insert(ids.begin(), npads[0], 1);
	std::vector<size_t> fakepds(npads[1], 1);
	std::vector<size_t> fakecds(npads[2], 1);
	fakepds.push_back(2); // ensures trimming never proceeds inward
	fakepds.insert(fakepds.end(), pds.begin(), pds.end());
	fakepds.push_back(2); // ensures trimming never proceeds inward
	fakepds.insert(fakepds.end(), npads[3], 1);
	fakecds.push_back(2); // ensures trimming never proceeds inward
	fakecds.insert(fakecds.end(), cds.begin(), cds.end());
	fakecds.push_back(2); // ensures trimming never proceeds inward
	fakecds.insert(fakecds.end(), npads[4], 1);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape fakeincom_ts(ids);
	nnet::tensorshape pcom_ts(fakepds);
	nnet::tensorshape com_ts(fakecds);

	EXPECT_EQ((size_t) 0, incom_ts.trim().rank());
	EXPECT_LT((size_t) 0, fakeincom_ts.rank());
	EXPECT_EQ((size_t) 0, fakeincom_ts.trim().rank());

	EXPECT_EQ(pds.size()+2, pcom_ts.trim().rank());
	ASSERT_EQ(cds.size()+2, com_ts.trim().rank());
}


// covers tensorshape: concatenate
TEST_F(TENSORSHAPE, Concat_A012)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	// undefined concatenating anything is that thing
	nnet::tensorshape none1 = incom_ts.concatenate(com_ts);
	nnet::tensorshape none2 = com_ts.concatenate(incom_ts);
	nnet::tensorshape none3 = incom_ts.concatenate(pcom_ts);
	nnet::tensorshape none4 = pcom_ts.concatenate(incom_ts);
	EXPECT_TRUE(tensorshape_equal(none1, com_ts)) << 
		testutils::sprintf("expecting %p, got %p", &none1, &com_ts);
	EXPECT_TRUE(tensorshape_equal(none2, com_ts)) << 
		testutils::sprintf("expecting %p, got %p", &none2, &com_ts);
	EXPECT_TRUE(tensorshape_equal(none3, pcom_ts)) << 
		testutils::sprintf("expecting %p, got %p", &none3, &pcom_ts);
	EXPECT_TRUE(tensorshape_equal(none4, pcom_ts)) << 
		testutils::sprintf("expecting %p, got %p", &none4, &pcom_ts);

	std::vector<size_t> straight = com_ts.concatenate(pcom_ts).as_list();
	std::vector<size_t> backcat = pcom_ts.concatenate(com_ts).as_list();

	ASSERT_EQ(straight.size(), backcat.size());
	ASSERT_EQ(straight.size(), cds.size() + pds.size());
	std::vector<size_t> expect_str8 = cds;
	std::vector<size_t> expect_revr = pds;
	expect_str8.insert(expect_str8.end(), pds.begin(), pds.end());
	expect_revr.insert(expect_revr.end(), cds.begin(), cds.end());
	EXPECT_TRUE(std::equal(straight.begin(),
		straight.end(), expect_str8.begin()));
	EXPECT_TRUE(std::equal(backcat.begin(),
		backcat.end(), expect_revr.begin()));
}


// covers tensorshape: 
// with_rank, with_rank_at_least, with_rank_at_most, depends on rank
TEST_F(TENSORSHAPE, WithRank_A013)
{
	std::vector<size_t> ids;
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	random_shapes(this, pds, cds);
	nnet::tensorshape incom_ts;
	nnet::tensorshape pcom_ts(pds);
	nnet::tensorshape com_ts(cds);

	// expand rank
	size_t peak = std::max(pds.size(), cds.size());
	size_t trough = std::min(pds.size(), cds.size());
	std::vector<size_t> bounds = get_int(2, "bounds", {1, trough});
	size_t upperbound = peak + bounds[0];
	size_t lowerbound = trough - bounds[1];
	// expansion
	EXPECT_EQ(upperbound, incom_ts.with_rank(upperbound).rank());
	EXPECT_EQ(upperbound, pcom_ts.with_rank(upperbound).rank());
	EXPECT_EQ(upperbound, com_ts.with_rank(upperbound).rank());
	// compression
	EXPECT_EQ(lowerbound, incom_ts.with_rank(lowerbound).rank());
	EXPECT_EQ(lowerbound, pcom_ts.with_rank(lowerbound).rank());
	EXPECT_EQ(lowerbound, com_ts.with_rank(lowerbound).rank());

	// favor higher dimensionalities
	EXPECT_EQ(upperbound, incom_ts.with_rank_at_least(upperbound).rank());
	EXPECT_EQ(upperbound, pcom_ts.with_rank_at_least(upperbound).rank());
	EXPECT_EQ(upperbound, com_ts.with_rank_at_least(upperbound).rank());
	EXPECT_EQ(lowerbound, incom_ts.with_rank_at_least(lowerbound).rank());
	EXPECT_EQ(pds.size(), pcom_ts.with_rank_at_least(lowerbound).rank());
	EXPECT_EQ(cds.size(), com_ts.with_rank_at_least(lowerbound).rank());

	// favor lower dimensionalities
	EXPECT_EQ((size_t) 0, incom_ts.with_rank_at_most(upperbound).rank());
	EXPECT_EQ(pds.size(), pcom_ts.with_rank_at_most(upperbound).rank());
	EXPECT_EQ(cds.size(), com_ts.with_rank_at_most(upperbound).rank());
	EXPECT_EQ((size_t) 0, incom_ts.with_rank_at_most(lowerbound).rank());
	EXPECT_EQ(lowerbound, pcom_ts.with_rank_at_most(lowerbound).rank());
	EXPECT_EQ(lowerbound, com_ts.with_rank_at_most(lowerbound).rank());
}


// covers tensorshape: coord_from_idx, flat_idx
TEST_F(TENSORSHAPE, CoordMap_A014)
{
	std::vector<size_t> slist = random_def_shape(this);
	nnet::tensorshape shape(slist);
	std::vector<size_t> coord;
	for (size_t i = 0; i < shape.n_elems(); ++i)
	{
		coord = shape.coord_from_idx(i);
		assert(coord.size() == slist.size());
		size_t accum = 1;
		size_t index = 0;
		for (size_t j = 0; j < coord.size(); ++j)
		{
			index += coord[j] * accum;
			accum *= slist[j];
		}
		size_t cindex = shape.flat_idx(coord);
		EXPECT_EQ(i, index);
		EXPECT_EQ(i, cindex);
	}
}


#endif /* DISABLE_SHAPE_TEST */


#endif /* DISABLE_TENSOR_MODULE_TESTS */
