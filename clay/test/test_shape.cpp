#ifndef DISABLE_CLAY_MODULE_TESTS

#include "gtest/gtest.h"

#include "testutil/fuzz.hpp"
#include "testutil/sgen.hpp"
#include "testutil/check.hpp"

#include "clay/shape.hpp"


#ifndef DISABLE_SHAPE_TEST


using namespace testutil;


class SHAPE : public fuzz_test {};


// covers vector constructor and assignment
// and as_list
TEST_F(SHAPE, Construct_A000)
{
	std::vector<size_t> empty;
	std::vector<size_t> slist = random_shape(this, {1, 8});
	std::vector<size_t> slist2 = random_shape(this, {1, 8});
	clay::Shape def;
	clay::Shape vec(slist);
	clay::Shape assign;

	ASSERT_ARREQ(empty, def.as_list());
	ASSERT_ARREQ(slist, vec.as_list());

	EXPECT_ARREQ(empty, assign.as_list());
	assign = slist2;
	EXPECT_ARREQ(slist2, assign.as_list());

	clay::Shape cp_assign;
	clay::Shape mv_assign;

	clay::Shape cp(vec);
	clay::Shape mv(std::move(vec));

	EXPECT_ARREQ(slist, cp.as_list());
	EXPECT_ARREQ(slist, mv.as_list());
	EXPECT_ARREQ(empty, vec.as_list());

	cp_assign = cp;
	mv_assign = std::move(mv);

	EXPECT_ARREQ(slist, cp_assign.as_list());
	EXPECT_ARREQ(slist, mv_assign.as_list());
	EXPECT_ARREQ(empty, mv.as_list());
}


// covers Shape: operator []
TEST_F(SHAPE, IndexAccessor_A001)
{
	std::vector<size_t> svec = random_def_shape(this);
	clay::Shape shape(svec);
	size_t dim = get_int(1, "dim", {0, shape.rank()-1})[0];
	EXPECT_EQ(svec[dim], shape[dim]);
	EXPECT_THROW(shape[shape.rank()], std::out_of_range);
}


// covers Shape: n_elems
TEST_F(SHAPE, NElems_A002)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);

	size_t expect_nelems = 1;
	for (size_t c : cds)
	{
		expect_nelems *= c;
	}

	EXPECT_EQ((size_t) 0, incom_ts.n_elems());
	EXPECT_EQ((size_t) 0, pcom_ts.n_elems());
	EXPECT_EQ(expect_nelems, com_ts.n_elems());
}


// covers Shape: n_elems
TEST_F(SHAPE, NKnowns_A003)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);

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

	EXPECT_EQ((size_t) 0, incom_ts.n_known());
	EXPECT_EQ(expect_nknown, pcom_ts.n_known());
	EXPECT_EQ(expect_nelems, com_ts.n_known());
}


// covers Shape: rank
TEST_F(SHAPE, Rank_A004)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);

	EXPECT_EQ((size_t) 0, incom_ts.rank());
	EXPECT_EQ(pds.size(), pcom_ts.rank());
	ASSERT_EQ(cds.size(), com_ts.rank());
}


// covers Shape: is_compatible_with
TEST_F(SHAPE, Compatible_A005)
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
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);
	clay::Shape pcom2_ts(pds2);
	clay::Shape com2_ts(cds2);

	for (clay::Shape* shape : {&incom_ts, &pcom_ts, &com_ts, &pcom2_ts, &com2_ts})
	{
		EXPECT_TRUE(incom_ts.is_compatible_with(*shape)) << 
			std::string(ioutil::Stream() << "expecting " <<
			shape->as_list() << " to be compatible with empty");
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
	clay::Shape fake_ps(cds_cpy);
	clay::Shape fake_ps2(cds2_cpy);
	clay::Shape bad_ps(cds_cpy2);
	clay::Shape bad_ps2(cds2_cpy2);
	clay::Shape bad_ps3(brank);

	// guarantees
	EXPECT_TRUE(fake_ps.is_compatible_with(fake_ps)) << 
		std::string(ioutil::Stream() << "expecting " << 
		fake_ps.as_list() << " to be compatible with itself");
	EXPECT_TRUE(fake_ps2.is_compatible_with(fake_ps2)) << 
		std::string(ioutil::Stream() << "expecting " <<
		fake_ps2.as_list() << " to be compatible with itself");
	EXPECT_TRUE(fake_ps.is_compatible_with(com_ts)) <<
		std::string(ioutil::Stream() << "expecting " <<
		com_ts.as_list() << " to be compatible with " << fake_ps.as_list());
	EXPECT_TRUE(fake_ps2.is_compatible_with(com2_ts)) << 
		std::string(ioutil::Stream() << "expecting " <<
		com2_ts.as_list() << " to be compatible with " << fake_ps2.as_list());
	EXPECT_FALSE(fake_ps.is_compatible_with(bad_ps)) << 
		std::string(ioutil::Stream() << "expecting " << 
		bad_ps.as_list() << " to be incompatible with " << fake_ps.as_list());
	EXPECT_FALSE(fake_ps2.is_compatible_with(bad_ps2)) << 
		std::string(ioutil::Stream() << "expecting " <<
		bad_ps2.as_list() << " to be incompatible with " << fake_ps2.as_list());
	EXPECT_FALSE(fake_ps2.is_compatible_with(bad_ps3)) << 
		std::string(ioutil::Stream() << "expecting " <<
		bad_ps3.as_list() << " to be incompatible with " << fake_ps2.as_list());

	// fully defined are not expected to be compatible with bad
	EXPECT_TRUE(com_ts.is_compatible_with(com_ts)) << 
		std::string(ioutil::Stream() << "expecting " << 
		com_ts.as_list() << " to be compatible with itself");
	EXPECT_FALSE(com_ts.is_compatible_with(bad_ps)) << 
		std::string(ioutil::Stream() << "expecting " <<
		bad_ps.as_list() << " to be incompatible with " << com_ts.as_list());
	ASSERT_FALSE(com_ts.is_compatible_with(bad_ps2)) << 
		std::string(ioutil::Stream() << "expecting " <<
		bad_ps2.as_list() << " incompatible with " << com_ts.as_list());
}


// covers Shape: is_part_defined
TEST_F(SHAPE, PartDef_A006)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	random_shapes(this, pds, cds);
	random_shapes(this, pds2, cds2);
	// define partial and complete shapes
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);
	clay::Shape pcom2_ts(pds2);
	clay::Shape com2_ts(cds2);

	ASSERT_FALSE(incom_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		incom_ts.as_list() << " to be undefined");
	ASSERT_TRUE(pcom_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		pcom_ts.as_list() << " to be partially defined");
	ASSERT_TRUE(pcom2_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		pcom2_ts.as_list() << " to be partially defined");
	ASSERT_TRUE(com_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		com_ts.as_list() << " to be partially defined");
	ASSERT_TRUE(com2_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		com2_ts.as_list() << " to be partially defined");
}


// covers Shape: 
// is_fully_defined and assert_is_fully_defined
TEST_F(SHAPE, FullDef_A007)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	random_shapes(this, pds, cds);
	random_shapes(this, pds2, cds2);
	// define partial and complete shapes
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape pcom2_ts(pds2);
	clay::Shape com_ts(cds);
	clay::Shape com2_ts(cds2);

	EXPECT_FALSE(incom_ts.is_fully_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		incom_ts.as_list() << " to not be fully defined");
	EXPECT_FALSE(pcom_ts.is_fully_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		pcom_ts.as_list() << " to not be fully defined");
	EXPECT_FALSE(pcom2_ts.is_fully_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		pcom2_ts.as_list() << " to not be fully defined");
	EXPECT_TRUE(com_ts.is_fully_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		com_ts.as_list() << " to be fully defined");
	EXPECT_TRUE(com2_ts.is_fully_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		com2_ts.as_list() << " to be fully defined");

	EXPECT_TRUE(com_ts.is_fully_defined()) << "com_ts is not fully defined";
	EXPECT_TRUE(com2_ts.is_fully_defined()) << "com2_ts is not fully defined";
}


// covers Shape: 
// undefine, dependent on is_part_defined
TEST_F(SHAPE, Undefine_A009)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);

	EXPECT_FALSE(incom_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		incom_ts.as_list() << " to be undefined");
	EXPECT_TRUE(pcom_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		pcom_ts.as_list() <<" to be partially defined");
	EXPECT_TRUE(com_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		com_ts.as_list() <<" to be partially defined");

	incom_ts.undefine();
	pcom_ts.undefine();
	com_ts.undefine();

	EXPECT_FALSE(incom_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		incom_ts.as_list() << " to be undefined");
	EXPECT_FALSE(pcom_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		pcom_ts.as_list() << " to be undefined");
	ASSERT_FALSE(com_ts.is_part_defined()) << 
		std::string(ioutil::Stream() << "expecting " <<
		com_ts.as_list() << " to be undefined");
}


// covers Shape: merge_with
TEST_F(SHAPE, Merge_A010)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	std::vector<size_t> pds2;
	std::vector<size_t> cds2;
	random_shapes(this, pds, cds);
	pds.push_back(1);
	random_shapes(this, pds2, cds2);
	// define partial and complete shapes
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);
	clay::Shape pcom2_ts(pds2);
	clay::Shape com2_ts(cds2);

	// incomplete shape can merge with anything
	for (clay::Shape* shape : {&pcom_ts, &com_ts, &pcom2_ts, &com2_ts, &incom_ts})
	{
		clay::Shape merged = incom_ts.merge_with(*shape);
		// we're expecting merged shape to be the same as input shape
		EXPECT_SHAPEQ((*shape), merged);
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
	clay::Shape fake_ps(cds_cpy);
	clay::Shape fake_ps2(cds2_cpy);
	clay::Shape incompatible(cds_cpy2);
	clay::Shape incompatible2(cds2_cpy2);

	clay::Shape merged = fake_ps.merge_with(com_ts);
	clay::Shape merged2 = fake_ps2.merge_with(com2_ts);
	EXPECT_SHAPEQ(merged,  com_ts);
	EXPECT_SHAPEQ(merged2,  com2_ts);
	clay::Shape merged3 = incompatible.merge_with(com_ts);
	clay::Shape merged4 = incompatible2.merge_with(com2_ts);
	EXPECT_SHAPEQ(incompatible, merged3);
	EXPECT_SHAPEQ(incompatible2, merged4);

	// merging different ranks will error
	ASSERT_GT(pds.size(), cds.size()); // true by generation implementation
	EXPECT_THROW(pcom_ts.merge_with(com_ts), std::exception);
}


// covers Shape:
// trim, dependent on rank
// trim, dependent on rank
TEST_F(SHAPE, Trim_A011)
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
	clay::Shape incom_ts;
	clay::Shape fakeincom_ts(ids);
	clay::Shape pcom_ts(fakepds);
	clay::Shape com_ts(fakecds);

	EXPECT_EQ((size_t) 0, incom_ts.trim().rank());
	EXPECT_LT((size_t) 0, fakeincom_ts.rank());
	EXPECT_EQ((size_t) 0, fakeincom_ts.trim().rank());

	EXPECT_EQ(pds.size()+2, pcom_ts.trim().rank());
	ASSERT_EQ(cds.size()+2, com_ts.trim().rank());
}


// covers Shape: concatenate
TEST_F(SHAPE, Concat_A012)
{
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	random_shapes(this, pds, cds);
	// define partial and complete shapes
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);

	// undefined concatenating anything is that thing
	clay::Shape none1 = incom_ts.concatenate(com_ts);
	clay::Shape none2 = com_ts.concatenate(incom_ts);
	clay::Shape none3 = incom_ts.concatenate(pcom_ts);
	clay::Shape none4 = pcom_ts.concatenate(incom_ts);
	EXPECT_SHAPEQ(none1,  com_ts);
	EXPECT_SHAPEQ(none2,  com_ts);
	EXPECT_SHAPEQ(none3,  pcom_ts);
	EXPECT_SHAPEQ(none4,  pcom_ts);

	std::vector<size_t> straight = com_ts.concatenate(pcom_ts).as_list();
	std::vector<size_t> backcat = pcom_ts.concatenate(com_ts).as_list();

	ASSERT_EQ(straight.size(), backcat.size());
	ASSERT_EQ(straight.size(), cds.size() + pds.size());
	std::vector<size_t> expect_str8 = cds;
	std::vector<size_t> expect_revr = pds;
	expect_str8.insert(expect_str8.end(), pds.begin(), pds.end());
	expect_revr.insert(expect_revr.end(), cds.begin(), cds.end());
	EXPECT_ARREQ(straight, expect_str8);
	EXPECT_ARREQ(backcat, expect_revr);
}


// covers Shape: 
// with_rank, with_rank_at_least, with_rank_at_most, depends on rank
TEST_F(SHAPE, WithRank_A013)
{
	std::vector<size_t> ids;
	std::vector<size_t> pds;
	std::vector<size_t> cds;
	// this generation is better for rank testing,
	// since pds and cds ranks are independent
	random_shapes(this, pds, cds);
	clay::Shape incom_ts;
	clay::Shape pcom_ts(pds);
	clay::Shape com_ts(cds);

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


// covers Shape: coord_from_idx, flat_idx
TEST_F(SHAPE, CoordMap_A014)
{
	std::vector<size_t> slist = random_def_shape(this);
	clay::Shape shape(slist);
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


#endif /* DISABLE_CLAY_MODULE_TESTS */
