#include "gtest/gtest.h"

#include "ade/test/common.hpp"

#include "ade/shape.hpp"


#ifndef DISABLE_SHAPE_TEST


struct SHAPE : public TestModel {};


TEST_F(SHAPE, Init)
{
	SESSION sess = get_session("SHAPE::Init");

	ade::Shape scalar;

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Shape vec(slist);
	uint8_t n = slist.size();

	std::vector<ade::DimT> longlist = get_longshape(sess, "n_longlist");
	ade::Shape lvec(longlist);

	std::vector<ade::DimT> zerolist = get_zeroshape(sess, "zerolist");
	EXPECT_THROW(ade::Shape junk(zerolist), std::exception);

	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, scalar.at(i));
	}

	for (uint8_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(slist[i], vec.at(i));
	}
	for (uint8_t i = n; i < ade::rank_cap; ++i)
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


TEST_F(SHAPE, VecAssign)
{
	SESSION sess = get_session("SHAPE::VecAssign");

	std::vector<ade::DimT> zerolist = get_zeroshape(sess, "zerolist");
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> junk = get_shape(sess, "junk");

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


TEST_F(SHAPE, Moves)
{
	SESSION sess = get_session("SHAPE::Moves");

	std::vector<ade::DimT> junk = get_shape(sess, "junk");
	std::vector<ade::DimT> slist = get_shape(sess, "slist");

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


TEST_F(SHAPE, NElems)
{
	SESSION sess = get_session("SHAPE::NElems");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Shape shape(slist);

	std::vector<ade::DimT> longlist = get_longshape(sess, "n_longlist");
	ade::Shape lshape(longlist);

	size_t expect_nelems = 1;
	for (ade::DimT c : slist)
	{
		expect_nelems *= c;
	}

	size_t expect_long_nelems = 1;
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		expect_long_nelems *= longlist[i];
	}

	EXPECT_EQ(expect_nelems, shape.n_elems());
	EXPECT_EQ(expect_long_nelems, lshape.n_elems());
	std::vector<int32_t> gotnelems = {(int32_t) shape.n_elems()};
	std::vector<int32_t> gotlnelems = {(int32_t) lshape.n_elems()};
}


TEST_F(SHAPE, NRank)
{
	SESSION sess = get_session("SHAPE::NRank");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Shape shape(slist);

	std::vector<ade::DimT> longlist = get_longshape(sess, "n_longlist");
	ade::Shape lshape(longlist);

	EXPECT_EQ(slist.size(), shape.n_rank());
	EXPECT_EQ(ade::rank_cap, lshape.n_rank());
}


TEST_F(SHAPE, Compatible)
{
	SESSION sess = get_session("SHAPE::Compatible");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Shape shape(slist);

	// shape is compatible with itself regardless of after idx
	for (uint8_t idx = 0; idx < ade::rank_cap; ++idx)
	{
		EXPECT_TRUE(shape.compatible_after(shape, idx)) <<
			"expect " << shape.to_string() <<
			" to be compatible with itself after idx " << unsigned(idx);
	}

	uint32_t insertion_pt = sess->get_scalar("insertion_pt",
		{0, (int32_t) slist.size()});
	std::vector<ade::DimT> ilist = slist;
	ilist.insert(ilist.begin() + insertion_pt, 2);
	ade::Shape ishape(ilist);
	for (uint8_t idx = 0, rank = shape.n_rank(); idx < rank; ++idx)
	{
		EXPECT_FALSE(shape.compatible_after(ishape, idx)) <<
			"expect " << shape.to_string() <<
			" to be incompatible with " << ishape.to_string() <<
			" after idx " << unsigned(idx);
	}

	ilist[insertion_pt] = 3;
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


TEST_F(SHAPE, ToString)
{
	SESSION sess = get_session("SHAPE::ToString");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	std::string out = shape.to_string();

	if (GENERATE_MODE)
	{
		sess->store_string("expect_out", out);
	}
	else
	{
		std::string expect_out = sess->expect_string("expect_out");
		EXPECT_STREQ(expect_out.c_str(), out.c_str());
	}
}


#endif /* DISABLE_SHAPE_TEST */
