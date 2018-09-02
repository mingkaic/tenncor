#include <sstream>

#include "gtest/gtest.h"

#include "ade/shape.hpp"


#ifndef DISABLE_SHAPE_TEST


template <typename Iterator>
std::string to_str (Iterator begin, Iterator end)
{
	std::stringstream ss;
	ss << "[";
	if (begin != end)
	{
		ss << *begin;
		for (++begin; begin != end; ++begin)
		{
			ss << "\\" << *begin;
		}
	}
	ss << "]";
	return ss.str();
}


#define EXPECT_ARREQ(arr, arr2)\
EXPECT_TRUE(std::equal(arr.begin(), arr.end(), arr2.begin())) <<\
"expect list " + to_str(arr.begin(), arr.end()) +\
", got " + to_str(arr2.begin(), arr2.end()) + " instead"


TEST(SHAPE, Init)
{
	ade::Shape scalar;

	std::vector<ade::DimT> slist = {2, 3};
	ade::Shape vec(slist);

	std::vector<ade::DimT> longlist = {1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17};
	ade::Shape lvec(longlist);

	std::vector<ade::DimT> zerolist = {1, 2, 0, 3};
	EXPECT_THROW(ade::Shape junk(zerolist), std::exception);

	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, scalar.at(i));
	}

	uint8_t nslist = slist.size();
	for (uint8_t i = 0; i < nslist; ++i)
	{
		EXPECT_EQ(slist[i], vec.at(i));
	}
	for (uint8_t i = nslist; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, vec.at(i));
	}

	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(longlist[i], lvec.at(i));
	}

	EXPECT_THROW(scalar.at(ade::rank_cap), std::out_of_range);
	EXPECT_THROW(vec.at(ade::rank_cap), std::out_of_range);
}


TEST(SHAPE, VecAssign)
{
	std::vector<ade::DimT> zerolist = {1, 2, 0, 3};
	std::vector<ade::DimT> junk = {1, 3, 3, 7};
	std::vector<ade::DimT> slist = {2, 3};

	ade::Shape vecassign;
	ade::Shape vecassign2(junk);

	vecassign = slist;
	EXPECT_EQ(slist.size(), vecassign.n_rank());
	EXPECT_ARREQ(slist, vecassign.as_list());

	vecassign2 = slist;
	EXPECT_EQ(slist.size(), vecassign2.n_rank());
	EXPECT_ARREQ(slist, vecassign2.as_list());

	EXPECT_THROW(vecassign = zerolist, std::exception);
}


TEST(SHAPE, Moves)
{
	std::vector<ade::DimT> junk = {1, 3, 3, 7};
	std::vector<ade::DimT> slist = {2, 3};

	ade::Shape mvassign;
	ade::Shape mvassign2(junk);
	ade::Shape orig(slist);

	ade::Shape mv(std::move(orig));
	EXPECT_EQ(slist.size(), mv.n_rank());
	EXPECT_ARREQ(slist, mv.as_list());
	EXPECT_EQ(0, orig.n_rank());
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, orig.at(i));
	}

	mvassign = std::move(mv);
	EXPECT_EQ(slist.size(), mvassign.n_rank());
	EXPECT_ARREQ(slist, mvassign.as_list());
	EXPECT_EQ(0, mv.n_rank());
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, mv.at(i));
	}

	mvassign2 = std::move(mvassign);
	EXPECT_EQ(slist.size(), mvassign2.n_rank());
	EXPECT_ARREQ(slist, mvassign2.as_list());
	EXPECT_EQ(0, mvassign.n_rank());
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, mvassign.at(i));
	}
}


TEST(SHAPE, NElems)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Shape shape(slist);

	std::vector<ade::DimT> longlist = {1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17};
	ade::Shape lshape(longlist);

	size_t expect_nelems = 1;
	for (ade::DimT c : slist)
	{
		expect_nelems *= c;
	}

	size_t expect_long = 1;
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		expect_long *= longlist[i];
	}

	EXPECT_EQ(expect_nelems, shape.n_elems());
	EXPECT_EQ(expect_long, lshape.n_elems());
}


TEST(SHAPE, NRank)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Shape shape(slist);

	std::vector<ade::DimT> longlist = {1, 2, 3, 4, 5, 6, 7, 8, 9,
		10, 11, 12, 13, 14, 15, 16, 17};
	ade::Shape lshape(longlist);

	uint8_t cap = ade::rank_cap;
	EXPECT_EQ(slist.size(), shape.n_rank());
	EXPECT_EQ(cap, lshape.n_rank());
}


TEST(SHAPE, Compatible)
{
	// assert slist.size() < 16
	std::vector<ade::DimT> slist = {2, 3};
	ade::Shape shape(slist);

	// shape is compatible with itself regardless of after idx
	for (uint8_t idx = 0; idx < ade::rank_cap; ++idx)
	{
		EXPECT_TRUE(shape.compatible_after(shape, idx)) <<
			"expect " << shape.to_string() <<
			" to be compatible with itself after idx " << unsigned(idx);
	}

	uint8_t insertion_pt = 1;
	std::vector<ade::DimT> ilist = slist;
	ilist.insert(ilist.begin() + insertion_pt, 1);
	ade::Shape ishape(ilist);
	for (uint8_t idx = 0, rank = ishape.n_rank(); idx < rank; ++idx)
	{
		EXPECT_FALSE(shape.compatible_after(ishape, idx)) <<
			"expect " << shape.to_string() <<
			" to be incompatible with " << ishape.to_string() <<
			" after idx " << unsigned(idx);
	}

	ilist[insertion_pt] = 2;
	ade::Shape ishape2(ilist);
	for (uint8_t idx = 0; idx <= insertion_pt; ++idx)
	{
		EXPECT_FALSE(ishape.compatible_after(ishape2, idx)) <<
			"expect " << ishape.to_string() <<
			" to be incompatible with " << ishape2.to_string() <<
			" after idx " << unsigned(idx);
	}
	for (uint8_t idx = insertion_pt + 1; idx < ade::rank_cap; ++idx)
	{
		EXPECT_TRUE(ishape.compatible_after(ishape2, idx)) <<
			"shape " << ishape.to_string() <<
			" to be compatible with " << ishape2.to_string() <<
			" after idx " << unsigned(idx);
	}
}


TEST(SHAPE, ToString)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Shape shape(slist);
	std::string out = shape.to_string();
	std::string expect_out = "[2\\3]";
	EXPECT_STREQ(expect_out.c_str(), out.c_str());
}


#endif /* DISABLE_SHAPE_TEST */
