#ifndef DISABLE_MODULE_TESTS

#include "gtest/gtest.h"

#include "fuzzutil/fuzz.hpp"
#include "fuzzutil/check.hpp"

#include "soil/shape.hpp"


#ifndef DISABLE_SHAPE_TEST


using namespace testutil;


class SHAPE : public fuzz_test {};


TEST_F(SHAPE, Init)
{
	Shape scalar;

	std::vector<DimT> slist = {2, 3}; // tie to fuzz engine
	Shape vec(slist);

	std::vector<DimT> longlist = {1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17}; // tie to fuzz engine
	Shape lvec(longlist);

	std::vector<DimT> zerolist = {1, 2, 0, 3}; // tie to fuzz engine
	EXPECT_THROW(Shape junk(zerolist), std::exception);

	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		EXPECT_EQ(1, scalar.at(i));
	}

	uint8_t nslist = slist.size();
	for (uint8_t i = 0; i < nslist; ++i)
	{
		EXPECT_EQ(slist[i], vec.at(i));
	}
	for (uint8_t i = nslist; i < rank_cap; ++i)
	{
		EXPECT_EQ(1, vec.at(i));
	}

	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		EXPECT_EQ(longlist[i], lvec.at(i));
	}

	EXPECT_THROW(scalar.at(rank_cap), std::out_of_range);
	EXPECT_THROW(vec.at(rank_cap), std::out_of_range);
}


TEST_F(SHAPE, Copies)
{
	Shape cpassign;
	Shape mvassign;
	Shape vecassign;

	std::vector<DimT> junk = {1, 3, 3, 7}; // tie to fuzz engine
	Shape cpassign2(junk);
	Shape mvassign2(junk);
	Shape vecassign2(junk);

	std::vector<DimT> slist = {2, 3}; // tie to fuzz engine
	std::vector<DimT> zerolist = {1, 2, 0, 3}; // tie to fuzz engine
	Shape orig(slist);

	Shape cp(orig);
	EXPECT_EQ(slist.size(), cp.n_rank());
	EXPECT_ARREQ(slist, cp.as_list());

	Shape mv(std::move(orig));
	EXPECT_EQ(slist.size(), mv.n_rank());
	EXPECT_ARREQ(slist, mv.as_list());
	EXPECT_EQ(0, orig.n_rank());
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		EXPECT_EQ(1, orig.at(i));
	}

	cpassign = cp;
	EXPECT_EQ(slist.size(), cpassign.n_rank());
	EXPECT_ARREQ(slist, cpassign.as_list());

	cpassign2 = cp;
	EXPECT_EQ(slist.size(), cpassign2.n_rank());
	EXPECT_ARREQ(slist, cpassign2.as_list());

	mvassign = std::move(mv);
	EXPECT_EQ(slist.size(), mvassign.n_rank());
	EXPECT_ARREQ(slist, mvassign.as_list());
	EXPECT_EQ(0, mv.n_rank());
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		EXPECT_EQ(1, mv.at(i));
	}

	mvassign2 = std::move(mvassign);
	EXPECT_EQ(slist.size(), mvassign2.n_rank());
	EXPECT_ARREQ(slist, mvassign2.as_list());
	EXPECT_EQ(0, mvassign.n_rank());
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		EXPECT_EQ(1, mvassign.at(i));
	}

	vecassign = slist;
	EXPECT_EQ(slist.size(), vecassign.n_rank());
	EXPECT_ARREQ(slist, vecassign.as_list());

	vecassign2 = slist;
	EXPECT_EQ(slist.size(), vecassign2.n_rank());
	EXPECT_ARREQ(slist, vecassign2.as_list());

	EXPECT_THROW(vecassign = zerolist, std::exception);
}


TEST_F(SHAPE, NElems)
{
	std::vector<DimT> slist = {2, 3}; // tie to fuzz engine
	Shape shape(slist);

	std::vector<DimT> longlist = {1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17}; // tie to fuzz engine
	Shape lshape(longlist);

	size_t expect_nelems = 1;
	for (DimT c : slist)
	{
		expect_nelems *= c;
	}

	size_t expect_long = 1;
	for (uint8_t i = 0; i < rank_cap; ++i)
	{
		expect_long *= longlist[i];
	}

	EXPECT_EQ(expect_nelems, shape.n_elems());
	EXPECT_EQ(expect_long, lshape.n_elems());
}


TEST_F(SHAPE, NRank)
{
	std::vector<DimT> slist = {2, 3}; // tie to fuzz engine
	Shape shape(slist);

	std::vector<DimT> longlist = {1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17}; // tie to fuzz engine
	Shape lshape(longlist);

	uint8_t cap = rank_cap;
	EXPECT_EQ(slist.size(), shape.n_rank());
	EXPECT_EQ(cap, lshape.n_rank());
}


TEST_F(SHAPE, Compatible)
{
	// assert slist.size() < 16
	std::vector<DimT> slist = {2, 3}; // tie to fuzz engine
	Shape shape(slist);

	// shape is compatible with itself regardless of after idx
	for (uint8_t idx = 0; idx < rank_cap; ++idx)
	{
		EXPECT_TRUE(shape.compatible_after(shape, idx)) <<
			"expect " << shape.to_string() <<
			" to be compatible with itself after idx " << unsigned(idx);
	}

	uint8_t insertion_pt = 1;
	std::vector<DimT> ilist = slist;
	ilist.insert(ilist.begin() + insertion_pt, 1);
	Shape ishape(ilist);
	for (uint8_t idx = 0, rank = ishape.n_rank(); idx < rank; ++idx)
	{
		EXPECT_FALSE(shape.compatible_after(ishape, idx)) <<
			"expect " << shape.to_string() <<
			" to be incompatible with " << ishape.to_string() <<
			" after idx " << unsigned(idx);
	}

	ilist[insertion_pt] = 2;
	Shape ishape2(ilist);
	for (uint8_t idx = 0; idx <= insertion_pt; ++idx)
	{
		EXPECT_FALSE(ishape.compatible_after(ishape2, idx)) <<
			"expect " << ishape.to_string() <<
			" to be incompatible with " << ishape2.to_string() <<
			" after idx " << unsigned(idx);
	}
	for (uint8_t idx = insertion_pt + 1; idx < rank_cap; ++idx)
	{
		EXPECT_TRUE(ishape.compatible_after(ishape2, idx)) <<
			"shape " << ishape.to_string() <<
			" to be compatible with " << ishape2.to_string() <<
			" after idx " << unsigned(idx);
	}
}


TEST_F(SHAPE, ToString)
{
	std::vector<DimT> slist = {2, 3}; // tie to fuzz engine
	Shape shape(slist);
	std::string out = shape.to_string();
	std::string expect_out = std::to_string(slist[0]);
	for (size_t i = 1, n = slist.size(); i < n; ++i)
	{
		expect_out += " " + std::to_string(slist[i]);
	}
	EXPECT_STREQ(expect_out.c_str(), out.c_str());
}


#endif /* DISABLE_SHAPE_TEST */


#endif /* DISABLE_MODULE_TESTS */
