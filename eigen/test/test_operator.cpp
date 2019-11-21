
#ifndef DISABLE_OPERATOR_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"

#include "eigen/operator.hpp"

#include "eigen/mock/edge.hpp"


static void test_reduce (
	std::function<eigen::EigenptrT<double>(const eigen::iEigenEdge<double>&)> red,
	std::function<double(double,double)> agg)
{
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({3, 2}))),
		std::vector<double>{2, 3, 4, 5, 6, 7},
		teq::Shape({3}),
		std::vector<double>{1});
	auto r = red(edge);

	double* raw = r->get_ptr();
	r->assign();
	EXPECT_EQ(agg(2, 5), raw[0]);
	EXPECT_EQ(agg(3, 6), raw[1]);
	EXPECT_EQ(agg(4, 7), raw[2]);
}


TEST(OPERATOR, ReduceSum)
{
	test_reduce(eigen::reduce_sum<double>, [](double a, double b) { return a + b; });
}


TEST(OPERATOR, ReduceProd)
{
	test_reduce(eigen::reduce_prod<double>, [](double a, double b) { return a * b; });
}


TEST(OPERATOR, ReduceMin)
{
	test_reduce(eigen::reduce_min<double>, [](double a, double b) { return std::min(a, b); });
}


TEST(OPERATOR, ReduceMax)
{
	test_reduce(eigen::reduce_max<double>, [](double a, double b) { return std::max(a, b); });
}


TEST(OPERATOR, ArgMax)
{
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({3, 2}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		teq::Shape({3}),
		std::vector<double>{1});
	auto r = eigen::argmax(edge);

	double* raw = r->get_ptr();
	r->assign();
	EXPECT_EQ(1, raw[0]);
	EXPECT_EQ(0, raw[1]);
	EXPECT_EQ(1, raw[2]);

	MockEdge<double> edge2(
		teq::TensptrT(new MockTensor(teq::Shape({3, 2}))),
		std::vector<double>{2, 8, 4, 5, 9, 7},
		teq::Shape({1}),
		std::vector<double>{8});
	auto r2 = eigen::argmax(edge2);

	double* raw2 = r2->get_ptr();
	r2->assign();
	EXPECT_EQ(4, raw2[0]);
}


TEST(OPERATOR, Extend)
{
	teq::Shape outshape({3, 4, 2});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({3, 1, 2}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape,
		std::vector<double>{1, 4});
	auto r = eigen::extend(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 4, 2, 8, 4, 2, 8, 4, 2, 8, 4,
		5, 6, 7, 5, 6, 7, 5, 6, 7, 5, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Permute)
{
}


TEST(OPERATOR, Reshape)
{
}


TEST(OPERATOR, Slice)
{
}


TEST(OPERATOR, GroupConcat)
{
}


TEST(OPERATOR, GroupSum)
{
}


TEST(OPERATOR, GroupProd)
{
}


TEST(OPERATOR, Pad)
{
}


TEST(OPERATOR, Stride)
{
}


TEST(OPERATOR, Scatter)
{
}


TEST(OPERATOR, Reverse)
{
}


TEST(OPERATOR, Concat)
{
}


#endif // DISABLE_OPERATOR_TEST
