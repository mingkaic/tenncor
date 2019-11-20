
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
std::cout << eigen::make_tensmap(edge.data(), edge.argshape()) << std::endl;
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
}


TEST(OPERATOR, Extend)
{
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
