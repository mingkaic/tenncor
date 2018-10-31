
#ifndef DISABLE_COORD_TEST


#include "gtest/gtest.h"

#include "ade/coord.hpp"

#include "testutil/common.hpp"


struct COORD : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
	}
};


TEST_F(COORD, Inverse)
{
	simple::SessionT sess = get_session("COORD::Inverse");

	ade::MatrixT out, in;
	std::vector<double> indata = sess->get_double("indata",
		ade::mat_dim * ade::mat_dim, {0.0001, 5});
	for (uint8_t i = 0; i < ade::mat_dim; ++i)
	{
		for (uint8_t j = 0; j < ade::mat_dim; ++j)
		{
			in[i][j] = indata[i * ade::mat_dim + j];
		}
	}

	ade::inverse(out, in);

	// expect matmul is identity
	for (uint8_t i = 0; i < ade::mat_dim; ++i)
	{
		for (uint8_t j = 0; j < ade::mat_dim; ++j)
		{
			double val = 0;
			for (uint8_t k = 0; k < ade::mat_dim; ++k)
			{
				val += out[i][k] * in[k][j];
			}
			if (i == j)
			{
				EXPECT_DOUBLE_EQ(1, std::round(val));
			}
			else
			{
				EXPECT_DOUBLE_EQ(0, std::round(val));
			}
		}
	}
}


TEST_F(COORD, Identity)
{
	simple::SessionT sess = get_session("COORD::Identity");

	std::string idstr;
	ade::identity->access(
		[&](const ade::MatrixT& mat)
		{
			idstr = ade::to_string(mat);
		});

	EXPECT_STREQ("[[1\\0\\0\\0\\0\\0\\0\\0\\0]\\\n"
		"[0\\1\\0\\0\\0\\0\\0\\0\\0]\\\n"
		"[0\\0\\1\\0\\0\\0\\0\\0\\0]\\\n"
		"[0\\0\\0\\1\\0\\0\\0\\0\\0]\\\n"
		"[0\\0\\0\\0\\1\\0\\0\\0\\0]\\\n"
		"[0\\0\\0\\0\\0\\1\\0\\0\\0]\\\n"
		"[0\\0\\0\\0\\0\\0\\1\\0\\0]\\\n"
		"[0\\0\\0\\0\\0\\0\\0\\1\\0]\\\n"
		"[0\\0\\0\\0\\0\\0\\0\\0\\1]]", idstr.c_str());

	std::vector<int32_t> icoord = sess->get_int("icoord", ade::rank_cap, {0, 255});
	ade::CoordT fwd_out, bwd_out, in;
	std::copy(icoord.begin(), icoord.end(), in.begin());

	ade::identity->forward(fwd_out.begin(), in.begin());
	EXPECT_ARREQ(icoord, fwd_out);

	ade::identity->backward(bwd_out.begin(), in.begin());
	EXPECT_ARREQ(icoord, bwd_out);
}


TEST_F(COORD, Reduce)
{
	simple::SessionT sess = get_session("COORD::Reduce");

	size_t rank = sess->get_scalar("rank", {0, ade::rank_cap - 2});
	std::vector<int32_t> red = sess->get_int("red",
		ade::rank_cap - rank, {1, 16});
	std::vector<ade::DimT> dred(red.begin(), red.end());
	ade::CoordPtrT reducer = ade::reduce(rank, dred);

	std::vector<int32_t> icoord = sess->get_int("icoord", ade::rank_cap, {0, 255});
	ade::CoordT fwd_out, bwd_out, in;
	std::copy(icoord.begin(), icoord.end(), in.begin());

	reducer->forward(fwd_out.begin(), in.begin());
	for (size_t i = 0; i < rank; ++i)
	{
		EXPECT_EQ(icoord[i], fwd_out[i]) << i;
	}
	for (size_t i = rank; i < ade::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(icoord[i] / red[i - rank], fwd_out[i]) << "red=" << red[i - rank] << ",i=" << i;
	}

	reducer->backward(bwd_out.begin(), in.begin());
	for (size_t i = 0; i < rank; ++i)
	{
		EXPECT_EQ(icoord[i], bwd_out[i]) << i;
	}
	for (size_t i = rank; i < ade::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(icoord[i] * red[i - rank], bwd_out[i]) << "red=" << red[i - rank] << ",i=" << i;
	}

	EXPECT_FATAL(ade::reduce(rank, {0}), "cannot reduce using zero dimensions [0]");

	std::string fatalmsg = ade::sprintf(
		"cannot reduce shape rank %d beyond rank_cap with n_red %d",
		rank + 1, red.size());
	EXPECT_FATAL(ade::reduce(rank + 1, dred), fatalmsg.c_str());
}


TEST_F(COORD, Extend)
{
	simple::SessionT sess = get_session("COORD::Extend");

	size_t rank = sess->get_scalar("rank", {0, ade::rank_cap - 2});
	std::vector<int32_t> ext = sess->get_int("ext",
		ade::rank_cap - rank, {1, 16});
	std::vector<ade::DimT> dext(ext.begin(), ext.end());
	ade::CoordPtrT extender = ade::extend(rank, dext);

	std::vector<int32_t> icoord = sess->get_int("icoord", ade::rank_cap, {0, 255});
	ade::CoordT fwd_out, bwd_out, in;
	std::copy(icoord.begin(), icoord.end(), in.begin());

	extender->forward(fwd_out.begin(), in.begin());
	for (size_t i = 0; i < rank; ++i)
	{
		EXPECT_EQ(icoord[i], fwd_out[i]) << i;
	}
	for (size_t i = rank; i < ade::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(icoord[i] * ext[i - rank], fwd_out[i]) << "ext=" << ext[i - rank] << ",i=" << i;
	}

	extender->backward(bwd_out.begin(), in.begin());
	for (size_t i = 0; i < rank; ++i)
	{
		EXPECT_EQ(icoord[i], bwd_out[i]) << i;
	}
	for (size_t i = rank; i < ade::rank_cap; ++i)
	{
		EXPECT_DOUBLE_EQ(icoord[i] / ext[i - rank], bwd_out[i]) << "ext=" << ext[i - rank] << ",i=" << i;
	}

	EXPECT_FATAL(ade::extend(rank, {0}), "cannot extend using zero dimensions [0]");

	std::string fatalmsg = ade::sprintf(
		"cannot extend shape rank %d beyond rank_cap with n_ext %d",
		rank + 1, ext.size());
	EXPECT_FATAL(ade::extend(rank + 1, dext), fatalmsg.c_str());
}


TEST_F(COORD, Permute)
{
	simple::SessionT sess = get_session("COORD::Permute");

	size_t rank = sess->get_scalar("rank", {0, ade::rank_cap - 1});
	std::vector<uint64_t> perm = sess->choose("perm", ade::rank_cap, rank);
	std::vector<ade::DimT> dperm(perm.begin(), perm.end());
	ade::CoordPtrT permuter = ade::permute(dperm);
	std::array<bool,ade::rank_cap> permed;
	permed.fill(false);
	for (uint64_t p : perm)
	{
		permed[p] = true;
	}
	for (size_t i = 0; i < ade::rank_cap; ++i)
	{
		if (false == permed[i])
		{
			perm.push_back(i);
		}
	}

	std::vector<int32_t> icoord = sess->get_int("icoord", ade::rank_cap, {0, 255});
	ade::CoordT fwd_out, bwd_out, in;
	std::copy(icoord.begin(), icoord.end(), in.begin());

	permuter->forward(fwd_out.begin(), in.begin());
	for (size_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(icoord[perm[i]], fwd_out[i]);
	}

	permuter->backward(bwd_out.begin(), in.begin());
	for (size_t i = 0; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(icoord[i], bwd_out[perm[i]]);
	}
}


TEST_F(COORD, Flip)
{
	simple::SessionT sess = get_session("COORD::Flip");

	size_t dim = sess->get_scalar("dim", {0, ade::rank_cap - 1});
	ade::CoordPtrT flipper = ade::flip(dim);

	std::vector<int32_t> icoord = sess->get_int("icoord", ade::rank_cap, {0, 255});
	ade::CoordT fwd_out, bwd_out, in;
	std::copy(icoord.begin(), icoord.end(), in.begin());

	flipper->forward(fwd_out.begin(), in.begin());
	for (size_t i = 0; i < dim; ++i)
	{
		EXPECT_EQ(icoord[i], fwd_out[i]) << i;
	}
	for (size_t i = dim + 1; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(icoord[i], fwd_out[i]) << i;
	}
	EXPECT_EQ(-icoord[dim]-1, fwd_out[dim]);

	flipper->backward(bwd_out.begin(), in.begin());
	for (size_t i = 0; i < dim; ++i)
	{
		EXPECT_EQ(icoord[i], bwd_out[i]) << i;
	}
	for (size_t i = dim + 1; i < ade::rank_cap; ++i)
	{
		EXPECT_EQ(icoord[i], bwd_out[i]) << i;
	}

	EXPECT_EQ(-icoord[dim]-1, bwd_out[dim]);
}


#endif // DISABLE_COORD_TEST
