
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
	{
		teq::Shape outshape({3, 2, 2});
		MockEdge<double> edge(
			teq::TensptrT(new MockTensor(teq::Shape({2, 2, 3}))),
			std::vector<double>{2, 8, 4, 5, 6, 7, 1, 0, 9, 11, 10, 12},
			outshape,
			std::vector<double>{2, 0, 1, 3, 4, 5, 6, 7});
		auto r = eigen::permute(edge);

		double* raw = r->get_ptr();
		r->assign();

		std::vector<double> expect_raw = {
			2, 6, 9, 8, 7, 11,
			4, 1, 10, 5, 0, 12,
		};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_ARREQ(expect_raw, got_raw);
	}
	{
		teq::Shape outshape({3, 2});
		MockEdge<double> edge(
			teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
			std::vector<double>{2, 8, 4, 5, 6, 7},
			outshape,
			std::vector<double>{1, 0, 2, 3, 4, 5, 6, 7});
		auto r = eigen::permute(edge);

		double* raw = r->get_ptr();
		r->assign();

		std::vector<double> expect_raw = {2, 4, 6, 8, 5, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_ARREQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, Reshape)
{
	teq::Shape outshape({1, 3, 2});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({2, 1, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape);
	auto r = eigen::reshape(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 4, 5, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Slice)
{
	teq::Shape outshape({2, 1});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({3, 2}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape, std::vector<double>{1, 2, 1, 1});
	auto r = eigen::slice(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {6, 7};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
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


TEST(OPERATOR, DecodePair)
{
	std::vector<double> a = {1, 2, 3, 4, 5, 6};
	std::vector<double> bad = {1, 2, 3, 4, 5};

	auto apairs = eigen::decode_pair<size_t>(a);
	EXPECT_EQ(3, apairs.size());
	for (size_t i = 0; i < 3; ++i)
	{
		auto apair = apairs[i];
		EXPECT_EQ(apair.first + 1, apair.second);
		EXPECT_EQ(i * 2 + 1, apair.first);
	}
	EXPECT_FATAL(eigen::decode_pair<size_t>(bad),
		"cannot decode odd vector [1\\2\\3\\4\\5] into vec of pairs");
}


#endif // DISABLE_OPERATOR_TEST
