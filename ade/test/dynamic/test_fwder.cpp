
#ifndef DISABLE_FWDER_TEST


#include "gtest/gtest.h"

#include "ade/fwder.hpp"

#include "testutil/common.hpp"


struct FWDER : public simple::TestModel
{
	virtual void TearDown (void)
	{
		simple::TestModel::TearDown();
		TestLogger::latest_warning_ = "";
		TestLogger::latest_error_ = "";
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
	ade::Shape extshape(extlist);
	ade::Shape badshape(badlist);
	ade::Tensorptr scalar = ade::Tensor::get(ade::Shape());
	ade::Tensorptr leaf = ade::Tensor::get(shape);
	ade::Tensorptr leaf1 = ade::Tensor::get(shape);
	ade::Tensorptr ext_leaf = ade::Tensor::get(extshape);
	ade::Tensorptr badleaf = ade::Tensor::get(badshape);

	ade::Shape same_shape = ade::forwarder<OP>({leaf, leaf1});
	EXPECT_ARREQ(slist, same_shape.as_list());

	std::string fatalmsg = "cannot " + ade::opname(OP) +
		" with incompatible shapes " + shape.to_string() +
		" and " + badshape.to_string();
	EXPECT_FATAL(ade::forwarder<OP>({leaf, badleaf}), fatalmsg.c_str())

	std::string fatalmsg2 = "cannot " + ade::opname(OP) +
		" with incompatible shapes " + badshape.to_string() +
		" and " + shape.to_string();
	EXPECT_FATAL(ade::forwarder<OP>({badleaf, leaf}), fatalmsg2.c_str())

	std::stringstream fatalss;
	fatalss << "cannot map coordinate of rank 0, requires at least rank " <<
		(int) shape.n_rank();
	EXPECT_FATAL(ade::forwarder<OP>({leaf, scalar}), fatalss.str().c_str())
	EXPECT_FATAL(ade::forwarder<OP>({scalar, leaf1}), fatalss.str().c_str())

	std::stringstream fatalss2;
	fatalss2 << "cannot map coordinate of rank " << (int) shape.n_rank() <<
		", requires at least rank " << (int) shape.n_rank() + 1;
	EXPECT_FATAL(ade::forwarder<OP>({leaf, ext_leaf}), fatalss2.str().c_str())
	EXPECT_FATAL(ade::forwarder<OP>({leaf1, ext_leaf}), fatalss2.str().c_str());
}


template <ade::OPCODE OP>
static void scalar (simple::SessionT& sess)
{
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Shape scal_shape = ade::forwarder<OP>({leaf});
	EXPECT_EQ(1, scal_shape.n_elems());

	std::string fatalmsg = "cannot " + ade::opname(OP) +
		" for non-single argument(s): using 2 argument(s)";
	EXPECT_FATAL(ade::forwarder<OP>({leaf, leaf1}), fatalmsg.c_str())
}


template <ade::OPCODE OP>
static void reduce (simple::SessionT& sess)
{
	int32_t n = sess->get_scalar("n_slist", {2, ade::rank_cap - 1});
	std::vector<ade::DimT> slist = get_shape_n(sess, n, "slist");
	uint8_t dim = 1;
	if (n > 2)
	{
		dim = sess->get_scalar("dim", {1, n - 1});
	}
	uint8_t beyonddim = sess->get_scalar("beyonddim", {8, 16});
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Shape scal_shape = ade::forwarder<OP,uint8_t>({leaf}, beyonddim);
	EXPECT_EQ(1, scal_shape.n_elems());

	ade::Shape shape = ade::forwarder<OP,uint8_t>({leaf}, 0);
	EXPECT_EQ(slist, shape.as_list());
	EXPECT_STREQ("reducing coordinates [:0] ... created useless node",
		TestLogger::latest_warning_.c_str());

	ade::Shape red_shape = ade::forwarder<OP,uint8_t>({leaf}, dim);
	auto rlist = red_shape.as_list();
	std::vector<ade::DimT> exlist(slist.begin() + dim,
		slist.begin() + slist.size());
	EXPECT_ARREQ(exlist, rlist);

	std::string fatalmsg = "cannot " + ade::opname(OP) +
		" for non-single argument(s): using 2 argument(s)";
	auto fail = [&](){ ade::forwarder<OP,uint8_t>({leaf, leaf1}, 5); };
	EXPECT_FATAL(fail(), fatalmsg.c_str())
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


#define FWD_REDUCE(CODE)\
TEST_F(FWDER, CODE) {\
	simple::SessionT sess = get_session("FWDER::" + std::string(#CODE));\
	reduce<ade::CODE>(sess); }


FWD_REDUCE(ARGMAX)
FWD_REDUCE(RMAX)
FWD_REDUCE(RSUM)


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

	auto fail = [&](){ ade::forwarder<ade::MATMUL,uint8_t,uint8_t>(
		{a},1,1); };
	EXPECT_FATAL(fail(), "cannot MATMUL without 2 arguments: using 1 argument(s)")

	ade::Shape mat2d = ade::forwarder<ade::MATMUL,uint8_t,uint8_t>({a, b}, 1, 1);

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
	ade::Shape ashape(alist);
	ade::Shape bshape(blist);
	ade::Tensorptr a = ade::Tensor::get(ashape);
	ade::Tensorptr b = ade::Tensor::get(bshape);

	std::vector<ade::DimT> badalist = bad_cgroup;
	badalist.insert(badalist.end(), agroup.begin(), agroup.end());
	std::vector<ade::DimT> badblist = bgroup;
	badblist.insert(badblist.end(), bad_cgroup.begin(), bad_cgroup.end());
	ade::Shape badashape(badalist);
	ade::Shape badbshape(badblist);
	ade::Tensorptr bad_a = ade::Tensor::get(badashape);
	ade::Tensorptr bad_b = ade::Tensor::get(badbshape);

	std::string fatalmsg = "incompatible common dimensions when matmuling shapes " +
		ashape.to_string() + ", " + badbshape.to_string();
	auto fail = [&](){ ade::forwarder<ade::MATMUL,uint8_t,uint8_t>(
		{a, bad_b}, common_group.size(), bgroup.size()); };
	EXPECT_FATAL(fail(), fatalmsg.c_str());

	std::string fatalmsg2 = "incompatible common dimensions when matmuling shapes " +
		badashape.to_string() + ", " + bshape.to_string();
	auto fail2 = [&](){ ade::forwarder<ade::MATMUL,uint8_t,uint8_t>(
		{bad_a, b}, common_group.size(), bgroup.size()); };
	EXPECT_FATAL(fail2(), fatalmsg2.c_str());

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

	auto fail = [&](){ ade::forwarder<ade::PERMUTE,std::vector<uint8_t>>(
		{leaf, leaf}, {}); };
	EXPECT_FATAL(fail(), "cannot PERMUTE non-single argument(s): using 2 argument(s)");

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

	ade::Shape shape(slist);
	ade::Tensorptr leaf = ade::Tensor::get(shape);
	ade::Tensorptr badleaf = ade::Tensor::get(ade::Shape(badlist));
	auto fail = [&](){ ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>(
		{leaf, leaf}, {1}); };
	auto fail1 = [&](){ ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>(
		{leaf}, {0}); };
	auto fail2 = [&](){ ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>(
		{leaf}, bad_ext); };
	EXPECT_FATAL(fail(), "cannot EXTEND non-single argument(s): using 2 argument(s)");
	std::string fatalmsg;
	{
		std::vector<ade::DimT> zlist = slist;
		zlist.push_back(0);
		fatalmsg = "cannot create shape with vector containing zero: " +
			ade::to_string(zlist);
	}
	EXPECT_FATAL(fail1(), fatalmsg.c_str());
	std::string fatalmsg2;
	{
		fatalmsg2 = "cannot EXTEND dimension beyond rank_cap using vector " +
			ade::to_string(bad_ext) + " on shape " + shape.to_string();
	}
	EXPECT_FATAL(fail2(), fatalmsg2.c_str());
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


#endif // DISABLE_FWDER_TEST
