#include "gtest/gtest.h"

#include "ade/grader.hpp"


#ifndef DISABLE_GRADER_TEST


TEST(GRADER, ABS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::ABS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::ABS>({leaf}, leaf);
}


TEST(GRADER, NEG)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::NEG>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NEG>({leaf}, leaf);
}


TEST(GRADER, NOT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::NOT>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NOT>({leaf}, leaf);
}


TEST(GRADER, SIN)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::SIN>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::SIN>({leaf}, leaf);
}


TEST(GRADER, COS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::COS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::COS>({leaf}, leaf);
}


TEST(GRADER, TAN)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::TAN>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::TAN>({leaf}, leaf);
}


TEST(GRADER, EXP)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::EXP>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::EXP>({leaf}, leaf);
}


TEST(GRADER, LOG)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::LOG>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::LOG>({leaf}, leaf);
}


TEST(GRADER, SQRT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::SQRT>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::SQRT>({leaf}, leaf);
}


TEST(GRADER, ROUND)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::ROUND>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::ROUND>({leaf}, leaf);
}


TEST(GRADER, FLIP)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::FLIP>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::FLIP>({leaf}, leaf);
}


TEST(GRADER, POW)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::POW>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::POW>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::POW>({leaf, leaf}, leaf);
}


TEST(GRADER, ADD)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::ADD>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::ADD>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::ADD>({leaf, leaf}, leaf);
}


TEST(GRADER, SUB)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::SUB>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::SUB>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::SUB>({leaf, leaf}, leaf);
}


TEST(GRADER, MUL)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::MUL>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::MUL>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::MUL>({leaf, leaf}, leaf);
}


TEST(GRADER, DIV)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::DIV>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::DIV>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::DIV>({leaf, leaf}, leaf);
}


TEST(GRADER, EQ)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::EQ>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::EQ>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::EQ>({leaf, leaf}, leaf);
}


TEST(GRADER, NE)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::NE>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NE>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::NE>({leaf, leaf}, leaf);
}


TEST(GRADER, LT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::LT>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::LT>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::LT>({leaf, leaf}, leaf);
}


TEST(GRADER, GT)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::GT>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::GT>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::GT>({leaf, leaf}, leaf);
}


TEST(GRADER, BINO)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::BINO>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::BINO>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::BINO>({leaf, leaf}, leaf);
}


TEST(GRADER, UNIF)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::UNIF>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::UNIF>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::UNIF>({leaf, leaf}, leaf);
}


TEST(GRADER, NORM)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g = ade::grader<ade::NORM>({leaf, leaf1}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::NORM>({leaf, leaf1}, leaf);
	ade::Tensorptr g2 = ade::grader<ade::NORM>({leaf, leaf}, leaf);
}


TEST(GRADER, N_ELEMS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::N_ELEMS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::N_ELEMS>({leaf}, leaf);
}


TEST(GRADER, N_DIMS)
{
	std::vector<ade::DimT> slist = {2, 3};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::N_DIMS>({leaf}, leaf1);
	ade::Tensorptr g1 = ade::grader<ade::N_DIMS>({leaf}, leaf);
}


TEST(GRADER, ARGMAX)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	EXPECT_THROW(ade::grader<ade::ARGMAX>({leaf}, leaf1), std::bad_function_call);
	EXPECT_THROW(ade::grader<ade::ARGMAX>({leaf}, leaf), std::bad_function_call);
}


TEST(GRADER, RMAX)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0_0 = ade::grader<ade::RMAX>({leaf}, leaf1);
	ade::Tensorptr g1_0 = ade::grader<ade::RMAX>({leaf}, leaf);
}


TEST(GRADER, RSUM)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0_0 = ade::grader<ade::RSUM>({leaf}, leaf1);
	ade::Tensorptr g1_0 = ade::grader<ade::RSUM>({leaf}, leaf);
}


TEST(GRADER, MATMUL)
{
	std::vector<ade::DimT> alist = {2, 3};
	std::vector<ade::DimT> blist = {4, 2};
	ade::Tensorptr a = ade::Tensor::get(ade::Shape(alist));
	ade::Tensorptr b = ade::Tensor::get(ade::Shape(blist));

	std::vector<ade::DimT> alist1 = {4, 2, 3};
	std::vector<ade::DimT> blist1 = {3, 4, 2};
	ade::Tensorptr a1 = ade::Tensor::get(ade::Shape(alist1));
	ade::Tensorptr b1 = ade::Tensor::get(ade::Shape(blist1));

	ade::Tensorptr ga = ade::grader<ade::MATMUL>({a, b}, a);
	ade::Tensorptr gb = ade::grader<ade::MATMUL>({a, b}, b);
	ade::Tensorptr z = ade::grader<ade::MATMUL>({a, b}, a1);
	ade::Tensorptr z1 = ade::grader<ade::MATMUL>({a, b}, b1);

	ade::Tensorptr g1a1 = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({a1, b1}, a1, 2, 1);
	ade::Tensorptr g1b1 = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({a1, b1}, b1, 2, 1);
	ade::Tensorptr g2a = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({a, a1}, a, 2, 1);
	ade::Tensorptr g2a1 = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({a, a1}, a1, 2, 1);
	ade::Tensorptr g3b = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({b, b1}, b, 2, 1);
	ade::Tensorptr g3b1 = ade::grader<
		ade::MATMUL,uint8_t,uint8_t>({b, b1}, b1, 2, 1);

	std::vector<ade::DimT> duplist = {3, 3};
	ade::Tensorptr d = ade::Tensor::get(ade::Shape(duplist));
	ade::Tensorptr gd = ade::grader<ade::MATMUL>({d, d}, d);
}


TEST(GRADER, PERMUTE)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};
	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, leaf1, {2, 1, 3, 4, 0});
	ade::Tensorptr g1 = ade::grader<ade::PERMUTE,
		std::vector<uint8_t>>({leaf}, leaf, {2, 1, 3, 4, 0});
}


TEST(GRADER, EXTEND)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::EXTEND,
		std::vector<ade::DimT>>({leaf}, leaf1, {4});
	ade::Tensorptr g1 = ade::grader<ade::EXTEND,
		std::vector<ade::DimT>>({leaf}, leaf, {4});
}


TEST(GRADER, RESHAPE)
{
	std::vector<ade::DimT> slist = {2, 3, 4, 5, 7};
	std::vector<ade::DimT> olist = {2, 12, 5, 7};

	ade::Tensorptr leaf = ade::Tensor::get(ade::Shape(slist));
	ade::Tensorptr leaf1 = ade::Tensor::get(ade::Shape(slist));

	ade::Tensorptr g0 = ade::grader<ade::RESHAPE,
		std::vector<ade::DimT>>({leaf}, leaf1, olist);
	ade::Tensorptr g1 = ade::grader<ade::RESHAPE,
		std::vector<ade::DimT>>({leaf}, leaf, olist);
}


#endif /* DISABLE_GRADER_TEST */
