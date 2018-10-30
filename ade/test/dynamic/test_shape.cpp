
#ifndef DISABLE_SHAPE_TEST


#include "gtest/gtest.h"

#include "ade/shape.hpp"

#include "testutil/common.hpp"


struct SHAPE : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(SHAPE, Init)
{
	simple::SessionT sess = get_session("SHAPE::Init");

	ade::Shape scalar;

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Shape vec(slist);
	uint8_t n = slist.size();

	std::vector<ade::DimT> longlist = get_longshape(sess, "n_longlist");
	ade::Shape lvec(longlist);

	std::vector<ade::DimT> zerolist = get_zeroshape(sess, "zerolist");
	std::string fatalmsg = "cannot create shape with vector containing zero: " +
		ade::to_string(zerolist.begin(), zerolist.end());
	EXPECT_FATAL(ade::Shape junk(zerolist), fatalmsg.c_str());

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
	simple::SessionT sess = get_session("SHAPE::VecAssign");

	std::vector<ade::DimT> zerolist = get_zeroshape(sess, "zerolist");
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> junk = get_shape(sess, "junk");

	ade::Shape vecassign;
	ade::Shape vecassign2(junk);

	vecassign = slist;
	std::vector<ade::DimT> vlist(vecassign.begin(), vecassign.end());
	EXPECT_ARREQ(slist, vlist);

	vecassign2 = slist;
	std::vector<ade::DimT> vlist2(vecassign2.begin(), vecassign2.end());
	EXPECT_ARREQ(slist, vlist2);

	std::string fatalmsg = "cannot create shape with vector containing zero: " +
		ade::to_string(zerolist.begin(), zerolist.end());
	EXPECT_FATAL(vecassign = zerolist, fatalmsg.c_str());
}


TEST_F(SHAPE, Moves)
{
	simple::SessionT sess = get_session("SHAPE::Moves");

	std::vector<ade::DimT> junk = get_shape(sess, "junk");
	std::vector<ade::DimT> slist = get_shape(sess, "slist");

	ade::Shape mvassign;
	ade::Shape mvassign2(junk);
	ade::Shape orig(slist);

	ade::Shape mv(std::move(orig));
	std::vector<ade::DimT> mlist(mv.begin(), mv.end());
	EXPECT_ARREQ(slist, mlist);
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, orig.at(i));
	}

	mvassign = std::move(mv);
	std::vector<ade::DimT> alist(mvassign.begin(), mvassign.end());
	EXPECT_ARREQ(slist, alist);
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, mv.at(i));
	}

	mvassign2 = std::move(mvassign);
	std::vector<ade::DimT> alist2(mvassign2.begin(), mvassign2.end());
	EXPECT_ARREQ(slist, alist2);
	for (uint8_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(1, mvassign.at(i));
	}
}


TEST_F(SHAPE, NElems)
{
	simple::SessionT sess = get_session("SHAPE::NElems");

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


TEST_F(SHAPE, Compatible)
{
	simple::SessionT sess = get_session("SHAPE::Compatible");

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
	for (uint8_t idx = 0; idx < insertion_pt; ++idx)
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
	simple::SessionT sess = get_session("SHAPE::ToString");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Shape shape(slist);
	std::string out = shape.to_string();

	optional<std::string> expect_out = sess->expect_string("expect_out");
	if (expect_out)
	{
		EXPECT_STREQ(expect_out->c_str(), out.c_str());
	}
	sess->store_string("expect_out", out);
}


#endif // DISABLE_SHAPE_TEST
