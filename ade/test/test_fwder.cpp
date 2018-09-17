#include "gtest/gtest.h"

#include "ade/test/common.hpp"

#include "ade/fwder.hpp"


#ifndef DISABLE_FWDER_TEST


struct FWDER : public TestModel {};


template <ade::OPCODE opcode>
static void unary_elementary (SESSION& sess)
{
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));

	ade::Shape same_shape = ade::forwarder<opcode>({leaf});
	EXPECT_ARREQ(slist, same_shape.as_list());
}


template <ade::OPCODE opcode>
static void binary_elementary (SESSION& sess)
{
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	int32_t incr_pt = 0;
	if (slist.size() > 1)
	{
		incr_pt = sess->get_scalar("incr_pt", {0, (int32_t) slist.size() - 1});
	}
	int32_t ext_value = sess->get_scalar("ext_value", {1, 13});
	std::vector<ade::DimT> badlist = slist;
	badlist[incr_pt]++;
	std::vector<ade::DimT> extlist = slist;
	extlist.push_back(ext_value);
	ade::Tensorptr scalar = ade::Tensor::get(ade::Shape());
	ade::Tensorptr alt_scalar = ade::Tensor::get(ade::Shape({1}));
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf2 = ade::Tensor::get(ade::Shape(extlist));
	ade::Tensorptr badleaf = ade::Tensor::get(ade::Shape(badlist));

	ade::Shape same_shape = ade::forwarder<opcode>({leaf, leaf1});
	ade::Shape same_shape2 = ade::forwarder<opcode>({leaf, scalar});
	ade::Shape same_shape3 = ade::forwarder<opcode>({scalar, leaf1});
	ade::Shape same_shape4 = ade::forwarder<opcode>({leaf, alt_scalar});
	ade::Shape same_shape5 = ade::forwarder<opcode>({alt_scalar, leaf1});
	EXPECT_ARREQ(slist, same_shape.as_list());
	EXPECT_ARREQ(slist, same_shape2.as_list());
	EXPECT_ARREQ(slist, same_shape3.as_list());
	EXPECT_ARREQ(slist, same_shape4.as_list());
	EXPECT_ARREQ(slist, same_shape5.as_list());

	ade::Shape ext_shape = ade::forwarder<opcode>({leaf, leaf2});
	ade::Shape ext_shape1 = ade::forwarder<opcode>({leaf1, leaf2});
	EXPECT_ARREQ(extlist, ext_shape.as_list());
	EXPECT_ARREQ(extlist, ext_shape1.as_list());

	EXPECT_THROW(ade::forwarder<opcode>({leaf, badleaf}), std::runtime_error) <<
		"leaf=" << leaf->shape().to_string() << ", badleaf=" << badleaf->shape().to_string();
	EXPECT_THROW(ade::forwarder<opcode>({badleaf, leaf}), std::runtime_error) <<
		"badleaf=" << badleaf->shape().to_string() << ", leaf=" << leaf->shape().to_string();
}


template <ade::OPCODE opcode>
static void scalar (SESSION& sess)
{
	std::vector<ade::DimT> slist = get_shape(sess, "slist");
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Shape scal_shape = ade::forwarder<opcode>({leaf});
	EXPECT_EQ(1, scal_shape.n_elems());

	EXPECT_THROW(ade::forwarder<opcode>({leaf, leaf1}), std::runtime_error);
}


#define FWD_UNAR(CODE)\
TEST_F(FWDER, CODE) {\
	SESSION sess = get_session("FWDER::" + std::string(#CODE));\
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
	SESSION sess = get_session("FWDER::" + std::string(#CODE));\
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

FWD_BINAR(BINO)
FWD_BINAR(UNIF)
FWD_BINAR(NORM)


#define FWD_SCALAR(CODE)\
TEST_F(FWDER, CODE) {\
	SESSION sess = get_session("FWDER::" + std::string(#CODE));\
	scalar<ade::CODE>(sess); }


FWD_SCALAR(N_ELEMS)
FWD_SCALAR(N_DIMS)
FWD_SCALAR(ARGMAX)
FWD_SCALAR(RMAX)
FWD_SCALAR(RSUM)


TEST_F(FWDER, MATMUL)
{
	std::vector<ade::DimT> alist = {2, 3};
	std::vector<ade::DimT> blist = {4, 2};
	ade::Tensorptr a = ade::Tensor::get(ade::Shape(alist));
	ade::Tensorptr b = ade::Tensor::get(ade::Shape(blist));

	std::vector<ade::DimT> alist1 = {4, 2, 3};
	std::vector<ade::DimT> blist1 = {3, 4, 2};
	ade::Tensorptr a1 = ade::Tensor::get(ade::Shape(alist1));
	ade::Tensorptr b1 = ade::Tensor::get(ade::Shape(blist1));

	ade::Shape m43 = ade::forwarder<ade::MATMUL>({a, b});
	EXPECT_EQ(2, m43.n_rank());
	EXPECT_EQ(blist[0], m43.at(0));
	EXPECT_EQ(alist[1], m43.at(1));

	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a}), std::runtime_error);
	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a1, b}), std::runtime_error);
	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a, b1}), std::runtime_error);
	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a1, b1}), std::runtime_error);

	ade::Shape m33 = ade::forwarder<
		ade::MATMUL,uint8_t,uint8_t>({a1, b1}, 2, 1);
	EXPECT_EQ(2, m33.n_rank());
	EXPECT_EQ(blist1[0], m33.at(0));
	EXPECT_EQ(alist1[2], m33.at(1));

	ade::Shape m32 = ade::forwarder<
		ade::MATMUL,uint8_t,uint8_t>({a, a1}, 2, 1);
	EXPECT_EQ(1, m32.n_rank());
	EXPECT_EQ(alist1[0], m32.at(0));

	ade::Shape m32a = ade::forwarder<
		ade::MATMUL,uint8_t,uint8_t>({b, b1}, 2, 1);
	EXPECT_EQ(1, m32a.n_rank());
	EXPECT_EQ(blist1[0], m32a.at(0));

	auto fail = [&]()
	{
		ade::forwarder<ade::MATMUL,uint8_t,uint8_t>({a1, b}, 2, 1);
	};
	auto fail1 = [&]()
	{
		ade::forwarder<ade::MATMUL,uint8_t,uint8_t>({a1, b1}, 0, 1);
	};
	auto fail2 = [&]()
	{
		ade::forwarder<ade::MATMUL,uint8_t,uint8_t>({a1, b1}, 1, 0);
	};
	auto fail3 = [&]()
	{
		ade::forwarder<ade::MATMUL,uint8_t,uint8_t>({a1, b1}, 7, 1);
	};
	auto fail4 = [&]()
	{
		ade::forwarder<ade::MATMUL,uint8_t,uint8_t>({a1, b1}, 1, 7);
	};
	EXPECT_THROW(fail(), std::runtime_error);
	EXPECT_THROW(fail1(), std::runtime_error);
	EXPECT_THROW(fail2(), std::runtime_error);
	EXPECT_THROW(fail3(), std::runtime_error);
	EXPECT_THROW(fail4(), std::runtime_error);
}


TEST_F(FWDER, PERMUTE)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));

	auto fail = [&]()
	{
		ade::forwarder<ade::PERMUTE,std::vector<uint8_t>>({leaf, leaf}, {});
	};
	EXPECT_THROW(fail(), std::runtime_error);

	std::vector<uint8_t> pidx = {2, 1, 3, 4, 0};
	ade::Shape perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, pidx);

	auto plist = perm.as_list();
	size_t np = plist.size();
	ASSERT_EQ(pidx.size(), np);
	// todo: improve this test (it's technically following the definition of the function)
	for (size_t i = 0; i < np; ++i)
	{
		EXPECT_EQ((int) slist[pidx[i]], (int) plist[i]) << "index " << i;
	}

	std::vector<uint8_t> pidx2 = {1, 4, 3, 0};
	std::vector<uint8_t> remaining_idx = {2};
	ade::Shape low_perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, pidx2);

	auto plist2 = low_perm.as_list();
	size_t np2 = plist2.size();
	ASSERT_EQ(slist.size(), np2);
	size_t npidx2 = pidx2.size();
	size_t nremaining = remaining_idx.size();
	ASSERT_EQ(npidx2 + nremaining, np2);
	for (size_t i = 0; i < npidx2; ++i)
	{
		EXPECT_EQ((int) slist[pidx2[i]], (int) plist2[i]) << "index " << i;
	}
	for (size_t i = 0; i < nremaining; ++i)
	{
		EXPECT_EQ((int) slist[remaining_idx[i]], (int) plist2[npidx2 + i]) << "index " << npidx2 + i;
	}

	std::vector<uint8_t> pidx3 = {2, 3, 3, 4, 1, 0};
	ade::Shape rep_perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, pidx3);

	auto plist3 = rep_perm.as_list();
	size_t np3 = plist3.size();
	ASSERT_EQ(pidx3.size(), np3);
	for (size_t i = 0; i < np3; ++i)
	{
		EXPECT_EQ((int) slist[pidx3[i]], (int) plist3[i]) << "index " << i;
	}
}


TEST_F(FWDER, EXTEND)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};
	std::vector<ade::DimT> badlist = {2, 3, 4, 5, 7, 1, 8};

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
		ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>({badleaf}, {4, 3});
	};
	auto fail3 = [&]()
	{
		ade::forwarder<ade::EXTEND,std::vector<ade::DimT>>({leaf}, {});
	};
	EXPECT_THROW(fail(), std::runtime_error);
	EXPECT_THROW(fail1(), std::runtime_error);
	EXPECT_THROW(fail2(), std::runtime_error);
	EXPECT_THROW(fail3(), std::runtime_error);

	std::vector<ade::DimT> ext = {4, 3};
	ade::Shape copied = ade::forwarder<ade::EXTEND,
		std::vector<ade::DimT>>({leaf}, ext);
	std::vector<ade::DimT> expect = slist;
	expect.insert(expect.end(), ext.begin(), ext.end());

	auto got = copied.as_list();
	EXPECT_ARREQ(expect, got);
}


TEST_F(FWDER, RESHAPE)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};
	std::vector<ade::DimT> olist = {2, 12, 5, 7};
	std::vector<ade::DimT> badlist = {2, 3, 4, 5, 7, 1, 8};

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


#endif /* DISABLE_FWDER_TEST */
