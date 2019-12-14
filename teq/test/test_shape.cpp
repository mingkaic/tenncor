
#ifndef DISABLE_SHAPE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/shape.hpp"


TEST(SHAPE, Init)
{
	teq::Shape scalar;

	std::vector<teq::DimT> slist = {12, 43, 56};
	teq::Shape vec(slist);
	teq::RankT n = slist.size();

	std::vector<teq::DimT> longlist = {4, 23, 44, 52, 19, 92, 12, 2, 5};
	teq::Shape lvec(longlist);

	std::vector<teq::DimT> zerolist = {43, 2, 5, 33, 0, 2, 7, 1};
	std::string fatalmsg = "cannot create shape with vector containing zero: " +
		fmts::to_string(zerolist.begin(), zerolist.end());
	EXPECT_FATAL(teq::Shape junk(zerolist), fatalmsg.c_str());

	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		EXPECT_EQ(1, scalar.at(i));
	}

	for (teq::RankT i = 0; i < n; ++i)
	{
		EXPECT_EQ(slist[i], vec.at(i));
	}
	for (teq::RankT i = n; i < teq::rank_cap; ++i)
	{
		EXPECT_EQ(1, vec.at(i));
	}

	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		EXPECT_EQ(longlist[i], lvec.at(i));
	}

	EXPECT_FATAL(scalar.at(teq::rank_cap), "cannot access out of bounds index 8");
	EXPECT_FATAL(vec.at(teq::rank_cap), "cannot access out of bounds index 8");
}


TEST(SHAPE, VecAssign)
{
	std::vector<teq::DimT> zerolist = {3, 0, 11, 89, 1, 1, 1, 1};
	std::vector<teq::DimT> slist = {52, 58, 35, 46, 77, 80};
	std::vector<teq::DimT> junk = {7, 42};

	teq::Shape vecassign;
	teq::Shape vecassign2(junk);

	vecassign = slist;
	std::vector<teq::DimT> vlist(vecassign.begin(), vecassign.end());
	EXPECT_ARREQ(slist, vlist);

	vecassign2 = slist;
	std::vector<teq::DimT> vlist2(vecassign2.begin(), vecassign2.end());
	EXPECT_ARREQ(slist, vlist2);

	std::string fatalmsg = "cannot create shape with vector containing zero: " +
		fmts::to_string(zerolist.begin(), zerolist.end());
	EXPECT_FATAL(vecassign = zerolist, fatalmsg.c_str());
}


TEST(SHAPE, Moves)
{
	std::vector<teq::DimT> junk = {8, 51, 73};
	std::vector<teq::DimT> slist = {24, 11, 12, 16};

	teq::Shape mvassign;
	teq::Shape mvassign2(junk);
	teq::Shape orig(slist);

	teq::Shape mv(std::move(orig));
	std::vector<teq::DimT> mlist(mv.begin(), mv.end());
	EXPECT_ARREQ(slist, mlist);

	mvassign = std::move(mv);
	std::vector<teq::DimT> alist(mvassign.begin(), mvassign.end());
	EXPECT_ARREQ(slist, alist);

	mvassign2 = std::move(mvassign);
	std::vector<teq::DimT> alist2(mvassign2.begin(), mvassign2.end());
	EXPECT_ARREQ(slist, alist2);
}


TEST(SHAPE, NElems)
{
	std::vector<teq::DimT> slist = {11, 12, 16};
	teq::Shape shape(slist);

	size_t expect_nelems = 11 * 12 * 16;
	EXPECT_EQ(expect_nelems, shape.n_elems());

	std::vector<teq::DimT> biglist(8, 255);
	teq::Shape bigshape(biglist);

	size_t expect_bignelems = 17878103347812890625ul;
	EXPECT_EQ(expect_bignelems, bigshape.n_elems());

	// also check the bounds
	EXPECT_GT(std::numeric_limits<teq::NElemT>::max(),
		expect_bignelems);
}


TEST(SHAPE, Compatible)
{
	std::vector<teq::DimT> slist = {20, 48, 10, 27, 65, 74};
	teq::Shape shape(slist);

	// shape is compatible with itself regardless of after idx
	for (teq::RankT idx = 0; idx < teq::rank_cap; ++idx)
	{
		EXPECT_TRUE(shape.compatible_after(shape, idx)) <<
			"expect " << shape.to_string() <<
			" to be compatible with itself after idx " << unsigned(idx);
	}

	uint32_t insertion_pt = 3;
	std::vector<teq::DimT> ilist = slist;
	ilist.insert(ilist.begin() + insertion_pt, 2);
	teq::Shape ishape(ilist);
	for (teq::RankT idx = 0; idx < insertion_pt; ++idx)
	{
		EXPECT_FALSE(shape.compatible_after(ishape, idx)) <<
			"expect " << shape.to_string() <<
			" to be incompatible with " << ishape.to_string() <<
			" after idx " << unsigned(idx);
	}

	ilist[insertion_pt] = 3;
	teq::Shape ishape2(ilist);
	for (teq::RankT idx = 0; idx <= insertion_pt; ++idx)
	{
		EXPECT_FALSE(ishape.compatible_after(ishape2, idx)) <<
			"expect " << ishape.to_string() <<
			" to be incompatible with " << ishape2.to_string() <<
			" after idx " << unsigned(idx);
	}
	for (teq::RankT idx = insertion_pt + 1; idx < teq::rank_cap; ++idx)
	{
		EXPECT_TRUE(ishape.compatible_after(ishape2, idx)) <<
			"shape " << ishape.to_string() <<
			" to be compatible with " << ishape2.to_string() <<
			" after idx " << unsigned(idx);
	}
}


TEST(SHAPE, Coordinates)
{
	std::vector<teq::DimT> slist = {9, 3, 7, 8, 5};
	teq::Shape shape(slist);
	teq::CoordT coord;
	for (teq::NElemT i = 0, n = shape.n_elems(); i < n; ++i)
	{
		coord = teq::coordinate(shape, i);
		for (teq::RankT i = 0; i < teq::rank_cap; ++i)
		{
			EXPECT_GT(shape.at(i), coord[i]);
		}
		teq::NElemT idx = teq::index(shape, coord);
		EXPECT_EQ(i, idx);
	}

	for (teq::RankT i = 0; i < teq::rank_cap; ++i)
	{
		coord[i] = shape.at(i);
	}
	std::string shapestr = shape.to_string();
	std::string fatalmsg = fmts::sprintf("cannot get index of bad coordinate "
		"%s for shape %s", shapestr.c_str(), shapestr.c_str());
	EXPECT_FATAL(teq::index(shape, coord), fatalmsg.c_str());

	std::string fatalmsg2 = fmts::sprintf("cannot get coordinate of index %d "
		"(>= shape %s)", shape.n_elems(), shapestr.c_str());
	EXPECT_FATAL(teq::coordinate(shape, shape.n_elems()), fatalmsg2.c_str());
}


TEST(SHAPE, ToString)
{
	std::vector<teq::DimT> slist = {24, 11, 12, 16, 7, 71, 1, 1};
	teq::Shape shape(slist);
	std::string out = shape.to_string();

	const char* expect_out = "[24\\11\\12\\16\\7\\71\\1\\1]";
	EXPECT_STREQ(expect_out, out.c_str());
}


#endif // DISABLE_SHAPE_TEST
