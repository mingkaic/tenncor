
#ifndef DISABLE_TEQ_SHAPE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/shape.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Throw;


struct SHAPE : public tutil::TestcaseWithLogger<> {};


TEST_F(SHAPE, Init)
{
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillRepeatedly(Return(true));

	teq::Shape scalar;

	teq::DimsT slist = {12, 43, 56};
	teq::Shape vec(slist);
	teq::RankT n = slist.size();

	teq::DimsT longlist = {4, 23, 44, 52, 19, 92, 12, 2, 5};
	teq::Shape lvec(longlist);

	teq::DimsT zerolist = {43, 2, 5, 33, 0, 2, 7};
	std::string fatalmsg = "cannot create shape with vector containing zero: " +
		fmts::to_string(zerolist.begin(), zerolist.end());
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg, _)).Times(1);
	teq::Shape junk(zerolist);

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

	EXPECT_EQ(0, scalar.n_ranks());
	EXPECT_EQ(slist.size(), vec.n_ranks());
	EXPECT_EQ(longlist.size(), lvec.n_ranks());
	EXPECT_EQ(1, scalar.at(teq::rank_cap));
	EXPECT_EQ(1, vec.at(teq::rank_cap));
	EXPECT_EQ(longlist.at(teq::rank_cap), lvec.at(teq::rank_cap));
}


TEST_F(SHAPE, Iterators)
{
	teq::Shape vec({12, 43, 56});
	auto it = vec.begin();
	auto et = vec.end();
	teq::RankT rank = vec.n_ranks();
	EXPECT_EQ(std::distance(it, et), rank);
	for (teq::RankT i = 0; i < rank; ++i)
	{
		EXPECT_NE(it + i, et);
	}

	{
		const teq::Shape cvec({12, 43, 56});
		auto cit = cvec.begin();
		auto cet = cvec.end();
		for (teq::RankT i = 0; i < cvec.n_ranks(); ++i)
		{
			EXPECT_NE(cit + i, cet);
		}
	}
}


TEST_F(SHAPE, VecAssign)
{
	EXPECT_CALL(*logger_, supports_level(logs::throw_err_level)).WillRepeatedly(Return(true));

	teq::DimsT zerolist = {3, 0, 11, 89};
	teq::DimsT slist = {52, 58, 35, 46, 77, 80};
	teq::DimsT junk = {7, 42};

	teq::Shape vecassign;
	teq::Shape vecassign2(junk);

	vecassign = slist;
	teq::DimsT vlist = vecassign.to_list();
	EXPECT_ARREQ(slist, vlist);

	vecassign2 = slist;
	teq::DimsT vlist2 = vecassign2.to_list();
	EXPECT_ARREQ(slist, vlist2);

	std::string fatalmsg = "cannot create shape with vector containing zero: " +
		fmts::to_string(zerolist.begin(), zerolist.end());
	EXPECT_CALL(*logger_, log(logs::throw_err_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(vecassign = zerolist, fatalmsg.c_str());
}


TEST_F(SHAPE, Moves)
{
	teq::DimsT junk = {8, 51, 73};
	teq::DimsT slist = {24, 11, 12, 16};

	teq::Shape mvassign;
	teq::Shape mvassign2(junk);
	teq::Shape orig(slist);

	teq::Shape mv(std::move(orig));
	teq::DimsT mlist = mv.to_list();
	EXPECT_ARREQ(slist, mlist);

	mvassign = std::move(mv);
	teq::DimsT alist = mvassign.to_list();
	EXPECT_ARREQ(slist, alist);

	mvassign2 = std::move(mvassign);
	teq::DimsT alist2 = mvassign2.to_list();
	EXPECT_ARREQ(slist, alist2);
}


TEST_F(SHAPE, NElems)
{
	teq::DimsT slist = {11, 12, 16};
	teq::Shape shape(slist);

	size_t expect_nelems = 11 * 12 * 16;
	EXPECT_EQ(expect_nelems, shape.n_elems());

	teq::DimsT biglist(8, 255);
	teq::Shape bigshape(biglist);

	size_t expect_bignelems = 17878103347812890625ul;
	EXPECT_EQ(expect_bignelems, bigshape.n_elems());

	// also check the bounds
	EXPECT_GT(std::numeric_limits<teq::NElemT>::max(),
		expect_bignelems);
}


TEST_F(SHAPE, Compatible)
{
	teq::DimsT slist = {20, 48, 10, 27, 65, 74};
	teq::Shape shape(slist);

	// shape is compatible with itself regardless of after idx
	for (teq::RankT idx = 0; idx < teq::rank_cap; ++idx)
	{
		EXPECT_TRUE(shape.compatible_after(shape, idx)) <<
			"expect " << shape.to_string() <<
			" to be compatible with itself after idx " << unsigned(idx);
	}

	uint32_t insertion_pt = 3;
	teq::DimsT ilist = slist;
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


TEST_F(SHAPE, ToString)
{
	teq::DimsT slist = {24, 11, 12, 16, 7, 71, 1, 1};
	teq::Shape shape(slist);
	std::string out = shape.to_string();

	const char* expect_out = "[24\\11\\12\\16\\7\\71\\1\\1]";
	EXPECT_STREQ(expect_out, out.c_str());
}


TEST_F(SHAPE, NarrowShape)
{
	teq::DimsT slist = {1, 2, 3, 4, 1};
	teq::DimsT elist = {1, 2, 3, 4};
	teq::Shape shape(slist);
	auto outs = teq::narrow_shape(shape);
	EXPECT_VECEQ(elist, outs);
}


#endif // DISABLE_TEQ_SHAPE_TEST
