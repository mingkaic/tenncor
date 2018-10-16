#include "gtest/gtest.h"

#include "ade/fwder.hpp"

#include "testutil/common.hpp"


#ifndef DISABLE_FWDER_TEST


struct FWDER : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
		TestLogger::latest_fatal_ = "";
	}
};


template <ade::OPCODE OP>
static void unary_elementary (simple::SessionT& sess)
{
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));

	ade::Shape same_shape = ade::forwarder<OP>({leaf});
	EXPECT_ARREQ(slist, same_shape.as_list());
}


template <ade::OPCODE OP>
static void binary_elementary (simple::SessionT& sess)
{
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> badlist = get_incompatible(sess, slist, "slist");
	int32_t ext_value = sess->get_scalar("ext_value", {2, 13});
	std::vector<ade::DimT> extlist = slist;
	extlist.push_back(ext_value);
	ade::Shape shape(slist);
	ade::Shape badshape(badlist);
	ade::Tensorptr scalar = ade::Tensor::get(ade::Shape());
	ade::Tensorptr alt_scalar = ade::Tensor::get(ade::Shape({1}));
	ade::Tensorptr leaf = ade::Tensor::get(shape);
	ade::Tensorptr leaf1 = ade::Tensor::get(shape);
	ade::Tensorptr leaf2 = ade::Tensor::get(ade::Shape(extlist));
	ade::Tensorptr badleaf = ade::Tensor::get(badshape);

	ade::Shape same_shape = ade::forwarder<OP>({leaf, leaf1});
	ade::Shape same_shape2 = ade::forwarder<OP>({leaf, scalar});
	ade::Shape same_shape3 = ade::forwarder<OP>({scalar, leaf1});
	ade::Shape same_shape4 = ade::forwarder<OP>({leaf, alt_scalar});
	ade::Shape same_shape5 = ade::forwarder<OP>({alt_scalar, leaf1});
	EXPECT_ARREQ(slist, same_shape.as_list());
	EXPECT_ARREQ(slist, same_shape2.as_list());
	EXPECT_ARREQ(slist, same_shape3.as_list());
	EXPECT_ARREQ(slist, same_shape4.as_list());
	EXPECT_ARREQ(slist, same_shape5.as_list());

	ade::Shape ext_shape = ade::forwarder<OP>({leaf, leaf2});
	ade::Shape ext_shape1 = ade::forwarder<OP>({leaf1, leaf2});
	EXPECT_ARREQ(extlist, ext_shape.as_list());
	EXPECT_ARREQ(extlist, ext_shape1.as_list());

	EXPECT_THROW(ade::forwarder<OP>({leaf, badleaf}), std::runtime_error) <<
		"leaf=" << leaf->shape().to_string() << ", badleaf=" << badleaf->shape().to_string();
	std::string expect_msg = "cannot " + ade::opname(OP) +
		" with incompatible shapes " + shape.to_string() + " and " + badshape.to_string();
	const char* fmsg = TestLogger::latest_fatal_.c_str();
	EXPECT_STREQ(expect_msg.c_str(), fmsg);
	EXPECT_THROW(ade::forwarder<OP>({badleaf, leaf}), std::runtime_error) <<
		"badleaf=" << badleaf->shape().to_string() << ", leaf=" << leaf->shape().to_string();
}


template <ade::OPCODE OP>
static void scalar (simple::SessionT& sess)
{
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Shape scal_shape = ade::forwarder<OP>({leaf});
	EXPECT_EQ(1, scal_shape.n_elems());

	EXPECT_THROW(ade::forwarder<OP>({leaf, leaf1}), std::runtime_error);
}


#define FWD_UNAR(CODE)\
TEST_F(FWDER, CODE) {\
	simple::SessionT sess = get_session("FWDER::" + std::string(#CODE));\
	unary_elementary<ade::CODE>(sess); }


FWD_UNAR(ABS)
FWD_UNAR(NEG)
FWD_UNAR(NOT)
FWD_UNAR(SIN)
FWD_UNAR(COS)
FWD_UNAR(TAN)
FWD_UNAR(EXP)
FWD_UNAR(LOG)
FWD_UNAR(SQRT)
FWD_UNAR(ROUND)
FWD_UNAR(FLIP)


#define FWD_BINAR(CODE)\
TEST_F(FWDER, CODE) {\
	simple::SessionT sess = get_session("FWDER::" + std::string(#CODE));\
	binary_elementary<ade::CODE>(sess); }


FWD_BINAR(POW)
FWD_BINAR(ADD)
FWD_BINAR(SUB)
FWD_BINAR(MUL)
FWD_BINAR(DIV)
FWD_BINAR(EQ)
FWD_BINAR(NE)
FWD_BINAR(LT)
FWD_BINAR(GT)

FWD_BINAR(RAND_BINO)
FWD_BINAR(RAND_UNIF)
FWD_BINAR(RAND_NORM)


#define FWD_SCALAR(CODE)\
TEST_F(FWDER, CODE) {\
	simple::SessionT sess = get_session("FWDER::" + std::string(#CODE));\
	scalar<ade::CODE>(sess); }


FWD_SCALAR(N_ELEMS)
FWD_SCALAR(N_DIMS)
FWD_SCALAR(ARGMAX)
FWD_SCALAR(RMAX)
FWD_SCALAR(RSUM)


TEST_F(FWDER, MATMUL2D)
{
	simple::SessionT sess = get_session("FWDER::MATMUL2D");

	ade::DimT cdim = sess->get_scalar("cdim", {1, 255});
	ade::DimT adim = sess->get_scalar("adim", {1, 255});
	ade::DimT bdim = sess->get_scalar("bdim", {1, 255});
	std::vector<ade::DimT> alist = {cdim, adim};
	std::vector<ade::DimT> blist = {bdim, cdim};
	ade::Tensorptr a = ade::Tensor::get(ade::Shape(alist));
	ade::Tensorptr b = ade::Tensor::get(ade::Shape(blist));

	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a}), std::runtime_error);

	ade::Shape mat2d = ade::forwarder<ade::MATMUL>({a, b});

	std::vector<ade::DimT> got = mat2d.as_list();
	int_verify(sess, "mat2d", std::vector<int32_t>(got.begin(), got.end()),
	[&]()
	{
		EXPECT_EQ(2, mat2d.n_rank());
		EXPECT_EQ(bdim, mat2d.at(0));
		EXPECT_EQ(adim, mat2d.at(1));
	});
}


TEST_F(FWDER, MATMUL)
{
	simple::SessionT sess = get_session("FWDER::MATMUL");

	ade::DimT agroupidx = sess->get_scalar("agroupidx", {1, ade::rank_cap - 1});
	std::vector<ade::DimT> common_group = get_shape_n(sess, agroupidx, "common_group");
	ade::DimT nremaining = ade::rank_cap - agroupidx;
	ade::DimT nagroup = 1;
	if (nremaining > 1)
	{
		nagroup = sess->get_scalar("agroupidx", {1, nremaining});
	}
	if (nagroup > agroupidx)
	{
		nremaining = ade::rank_cap - nagroup;
	}
	ade::DimT nbgroup = 1;
	if (nremaining > 1)
	{
		nbgroup = sess->get_scalar("bagroup", {1, nremaining});
	}
	std::vector<ade::DimT> agroup = get_shape_n(sess, nagroup, "agroup");
	std::vector<ade::DimT> bgroup = get_shape_n(sess, nbgroup, "bgroup");
	std::vector<ade::DimT> bad_cgroup = get_incompatible(sess, common_group, "common_group");

	std::vector<ade::DimT> alist = common_group;
	alist.insert(alist.end(), agroup.begin(), agroup.end());
	std::vector<ade::DimT> blist = bgroup;
	blist.insert(blist.end(), common_group.begin(), common_group.end());
	ade::Tensorptr a = ade::Tensor::get(ade::Shape(alist));
	ade::Tensorptr b = ade::Tensor::get(ade::Shape(blist));

	std::vector<ade::DimT> badalist = bad_cgroup;
	badalist.insert(badalist.end(), agroup.begin(), agroup.end());
	std::vector<ade::DimT> badblist = bgroup;
	badblist.insert(badblist.end(), bad_cgroup.begin(), bad_cgroup.end());
	ade::Tensorptr bad_a = ade::Tensor::get(ade::Shape(badalist));
	ade::Tensorptr bad_b = ade::Tensor::get(ade::Shape(badblist));

	auto fail = [&]()
	{
		ade::forwarder<ade::MATMUL,uint8_t,uint8_t>({a, bad_b}, common_group.size(), bgroup.size());
	};
	auto fail1 = [&]()
	{
		ade::forwarder<ade::MATMUL,uint8_t,uint8_t>({bad_a, b}, common_group.size(), bgroup.size());
	};

	EXPECT_THROW(fail(), std::runtime_error) <<
		"ashape=" << a->shape().to_string() << ", bad_bshape=" << bad_b->shape().to_string();
	EXPECT_THROW(fail1(), std::runtime_error) <<
		"bad_ashape=" << bad_a->shape().to_string() << ", bshape=" << b->shape().to_string();

	ade::Shape bigmat = ade::forwarder<
		ade::MATMUL,uint8_t,uint8_t>({a, b}, common_group.size(), bgroup.size());
	std::vector<ade::DimT> got = bigmat.as_list();
	int_verify(sess, "bigmat", std::vector<int32_t>(got.begin(), got.end()),
	[&]()
	{
		EXPECT_EQ(bgroup.size() + agroup.size(), (int) bigmat.n_rank());
		std::vector<ade::DimT> expect = bgroup;
		expect.insert(expect.end(), agroup.begin(), agroup.end());
		EXPECT_ARREQ(expect, got);
	});
}


TEST_F(FWDER, PERMUTE)
{
	simple::SessionT sess = get_session("FWDER::PERMUTE");

	int32_t n = sess->get_scalar("n", {2, ade::rank_cap - 2});
	std::vector<ade::DimT> slist = get_shape_n(sess, n, "slist");
	std::vector<uint64_t> pidx_temp = sess->choose("pidx", slist.size(), slist.size());
	std::vector<uint8_t> pidx(pidx_temp.begin(), pidx_temp.end());
	ade::Shape ogshape(slist);
	ade::Tensorptr leaf = ade::Tensor::get(ogshape);

	auto fail = [&]()
	{
		ade::forwarder<ade::PERMUTE,std::vector<uint8_t>>({leaf, leaf}, {});
	};
	EXPECT_THROW(fail(), std::runtime_error);

	ade::Shape perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, pidx);

	auto plist = perm.as_list();
	int_verify(sess, "plist", std::vector<int32_t>(plist.begin(), plist.end()),
	[&]()
	{
		size_t np = plist.size();
		ASSERT_EQ(pidx.size(), np);
		for (size_t i = 0; i < np; ++i)
		{
			EXPECT_EQ((int) ogshape.at(pidx[i]), (int) plist[i]) << "index " << i;
		}
	});

	int32_t divide = 1;
	if (n > 2)
	{
		divide = sess->get_scalar("divide", {1, n - 1});
	}
	auto sit = pidx.begin();
	auto ets = pidx.end();
	std::vector<uint8_t> pidx2(sit, sit + divide);
	std::vector<uint8_t> remaining_idx(sit + divide, ets);
	std::sort(remaining_idx.begin(), remaining_idx.end());
	ade::Shape low_perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, pidx2);

	auto plist2 = low_perm.as_list();
	int_verify(sess, "plist2", std::vector<int32_t>(plist2.begin(), plist2.end()),
	[&]()
	{
		size_t np2 = plist2.size();
		ASSERT_EQ(slist.size(), np2);
		size_t npidx2 = pidx2.size();
		size_t nremaining = remaining_idx.size();
		ASSERT_EQ(npidx2 + nremaining, np2);
		for (size_t i = 0; i < npidx2; ++i)
		{
			EXPECT_EQ((int) ogshape.at(pidx2[i]), (int) plist2[i]) << "index " << i;
		}
		for (size_t i = 0; i < nremaining; ++i)
		{
			EXPECT_EQ((int) ogshape.at(remaining_idx[i]), (int) plist2[npidx2 + i]) << "index " << npidx2 + i;
		}
	});

	int32_t naddition = sess->get_scalar("naddition", {1, ade::rank_cap - n});
	std::vector<int32_t> pidx3_temp = sess->get_int("pidx3", naddition, {0, ade::rank_cap - 1});
	std::vector<uint8_t> pidx3(pidx3_temp.begin(), pidx3_temp.end());
	pidx3.insert(pidx3.end(), pidx.begin(), pidx.end());
	ade::Shape rep_perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, pidx3);

	auto plist3 = rep_perm.as_list();
	int_verify(sess, "plist3", std::vector<int32_t>(plist3.begin(), plist3.end()),
	[&]()
	{
		size_t np3 = plist3.size();
		ASSERT_EQ(pidx3.size(), np3);
		for (size_t i = 0; i < np3; ++i)
		{
			EXPECT_EQ((int) ogshape.at(pidx3[i]), (int) plist3[i]) << "index " << i;
		}
	});
}


TEST_F(FWDER, EXTEND)
{
	simple::SessionT sess = get_session("FWDER::EXTEND");

	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> badlist = get_shape_n(sess, ade::rank_cap, "badlist");

	int32_t n = slist.size();
	int32_t remainder = ade::rank_cap - n;

	int32_t n_ext = 1;
	if (remainder > 1)
	{
		n_ext = sess->get_scalar("n_ext", {1, remainder});
	}
	std::vector<ade::DimT> ext = get_shape_n(sess, n_ext, "ext");
	std::vector<ade::DimT> bad_ext = get_shape_n(sess, remainder + 1, "bad_ext");

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr badleaf = ade::Tensor::get(ade::Shape(badlist));
	auto fail = [&]()
	{
		ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>({leaf, leaf}, {1});
	};
	auto fail1 = [&]()
	{
		ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>({leaf}, {0});
	};
	auto fail2 = [&]()
	{
		ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>({leaf}, bad_ext);
	};
	EXPECT_THROW(fail(), std::runtime_error);
	EXPECT_THROW(fail1(), std::runtime_error);
	EXPECT_THROW(fail2(), std::runtime_error);
	ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>({leaf}, {});
	EXPECT_STREQ("EXTENDing with empty vector... created useless node",
		TestLogger::latest_warning_.c_str());

	ade::Shape copied = ade::forwarder<ade::EXTEND,
		std::vector<ade::DimT>>({leaf}, ext);
	std::vector<ade::DimT> expect = slist;
	expect.insert(expect.end(), ext.begin(), ext.end());

	auto got = copied.as_list();
	EXPECT_ARREQ(expect, got);
}


TEST_F(FWDER, RESHAPE)
{
	simple::SessionT sess = get_session("FWDER::RESHAPE");

	int32_t n = sess->get_scalar("n", {2, ade::rank_cap - 2});
	std::vector<ade::DimT> slist = get_shape_n(sess, n, "slist");
	uint8_t mergeidx = 0;
	if (n > 2)
	{
		mergeidx = sess->get_scalar("mergeidx", {0, (uint8_t) slist.size() - 2});
	}
	std::vector<ade::DimT> olist = slist;
	olist.erase(olist.begin() + mergeidx);
	olist[mergeidx] *= slist[mergeidx];
	int32_t nremaining = ade::rank_cap - n;
	std::vector<ade::DimT> extra = get_shape_n(sess, nremaining, "extra");
	std::vector<ade::DimT> badlist = slist;
	badlist.insert(badlist.end(), extra.begin(), extra.end());

	ade::Tensorptr scalar = ade::Tensor::get(ade::Shape());
	ade::Tensorptr scalar1 = ade::Tensor::get(ade::Shape({1}));
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr badleaf = ade::Tensor::get(ade::Shape(badlist));
	auto fail = [&]()
	{
		ade::forwarder<ade::RESHAPE,std::vector<ade::DimT>>({leaf, leaf}, {1});
	};
	auto fail1 = [&]()
	{
		ade::forwarder<ade::RESHAPE,std::vector<ade::DimT>>({leaf}, {0});
	};
	auto fail2 = [&]()
	{
		ade::forwarder<ade::RESHAPE,std::vector<ade::DimT>>({leaf}, badlist);
	};
	EXPECT_THROW(fail(), std::runtime_error);
	EXPECT_THROW(fail1(), std::runtime_error);
	EXPECT_THROW(fail2(), std::runtime_error);

	ade::Shape res = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({leaf}, olist);
	std::vector<ade::DimT> got = res.as_list();
	EXPECT_ARREQ(olist, got);

	ade::Shape res1 = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({scalar}, slist);
	std::vector<ade::DimT> got1 = res1.as_list();
	EXPECT_ARREQ(slist, got1);

	ade::Shape res2 = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({scalar}, badlist);
	std::vector<ade::DimT> got2 = res2.as_list();
	EXPECT_ARREQ(badlist, got2);

	ade::Shape res3 = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({scalar1}, slist);
	std::vector<ade::DimT> got3 = res3.as_list();
	EXPECT_ARREQ(slist, got3);

	ade::Shape res4 = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({scalar1}, badlist);
	std::vector<ade::DimT> got4 = res4.as_list();
	EXPECT_ARREQ(badlist, got4);
}


#endif // DISABLE_FWDER_TEST
