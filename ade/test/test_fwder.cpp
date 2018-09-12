#include "gtest/gtest.h"

#include "ade/test/common.hpp"

#include "ade/fwder.hpp"


#ifndef DISABLE_FWDER_TEST


template <ade::OPCODE opcode>
static void unary_elementary (void)
{
	// SESSION sess = getSession("FWDER::" + ade::opname(opcode));

	// std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));

	ade::Shape same_shape = ade::forwarder<opcode>({leaf});
	EXPECT_ARREQ(slist, same_shape.as_list());
}


template <ade::OPCODE opcode>
static void binary_elementary (void)
{
	// SESSION sess = getSession("FWDER::" + ade::opname(opcode));

	// std::vector<ade::DimT> slist = get_shape(sess, "slist");
	// long incr_pt = sess->get_scalar("incr_pt", {0, slist.size()});
	// long ext_value = sess->get_scalar("ext_value");
	// std::vector<ade::DimT> badlist = slist;
	// badlist[insertion_pt]++;
	// std::vector<ade::DimT> extlist = slist;
	// extlist.push_back(ext_value);
	std::vector<ade::DimT> badlist = {2, 4};
	std::vector<ade::DimT> extlist = {2, 3, 4};
	std::vector<ade::DimT> slist = {2, 3};
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

	EXPECT_THROW(ade::forwarder<opcode>({leaf, badleaf}), std::runtime_error);
	EXPECT_THROW(ade::forwarder<opcode>({badleaf, leaf}), std::runtime_error);
}


template <ade::OPCODE opcode>
static void scalar (void)
{
	// SESSION sess = getSession("FWDER::" + ade::opname(opcode));

	// std::vector<ade::DimT> slist = get_shape(sess, "slist");
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Shape scal_shape = ade::forwarder<opcode>({leaf});
	EXPECT_EQ(1, scal_shape.n_elems());

	EXPECT_THROW(ade::forwarder<opcode>({leaf, leaf1}), std::runtime_error);
}


#define FWD_UNAR(CODE)\
TEST(FWDER, CODE) { unary_elementary<ade::CODE>(); }


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
TEST(FWDER, CODE) { binary_elementary<ade::CODE>(); }


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
TEST(FWDER, CODE) { scalar<ade::CODE>(); }


FWD_SCALAR(N_ELEMS)
FWD_SCALAR(N_DIMS)
FWD_SCALAR(ARGMAX)
FWD_SCALAR(RMAX)
FWD_SCALAR(RSUM)


TEST(FWDER, MATMUL)
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

	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a}), std::runtime_error);
	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a1, b}), std::runtime_error);
	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a, b1}), std::runtime_error);
	EXPECT_THROW(ade::forwarder<ade::MATMUL>({a1, b1}), std::runtime_error);

	ade::Shape m33 = ade::forwarder<
		ade::MATMUL,uint8_t,uint8_t>({a1, b1}, 2, 1);
	ade::Shape m32 = ade::forwarder<
		ade::MATMUL,uint8_t,uint8_t>({a, a1}, 2, 1);
	ade::Shape m32a = ade::forwarder<
		ade::MATMUL,uint8_t,uint8_t>({b, b1}, 2, 1);
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


TEST(FWDER, PERMUTE)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));

	auto fail = [&]()
	{
		ade::forwarder<ade::PERMUTE,std::vector<uint8_t>>({leaf, leaf}, {});
	};
	EXPECT_THROW(fail(), std::runtime_error);

	ade::Shape perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, {2, 1, 3, 4, 0});

	ade::Shape low_perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, {1, 4, 3, 0});

	ade::Shape rep_perm = ade::forwarder<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, {2, 3, 3, 4, 1, 0});
}


TEST(FWDER, EXTEND)
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

	ade::Shape copied = ade::forwarder<ade::EXTEND,
		std::vector<ade::DimT>>({leaf}, {4, 3});
}


TEST(FWDER, RESHAPE)
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
	ade::Shape res1 = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({scalar}, slist);
	ade::Shape res2 = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({scalar}, badlist);
	ade::Shape res3 = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({scalar1}, slist);
	ade::Shape res4 = ade::forwarder<ade::RESHAPE,
		std::vector<ade::DimT>>({scalar1}, badlist);
}


#endif /* DISABLE_FWDER_TEST */
