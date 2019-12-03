
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
	teq::TensptrT inarg(new MockTensor(teq::Shape({3, 2})));
	std::vector<double> indata = {2, 8, 4, 5, 6, 7};
	// slice both dimensions 0 and 1
	{
		teq::Shape outshape({2, 1});
		MockEdge<double> edge(inarg, indata,
			outshape, std::vector<double>{1, 2, 1, 1});
		auto r = eigen::slice(edge);

		double* raw = r->get_ptr();
		r->assign();

		std::vector<double> expect_raw = {6, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_ARREQ(expect_raw, got_raw);
	}
	// slice last dimension (validate optimization)
	{
		teq::Shape outshape({3, 1});
		MockEdge<double> edge(inarg, indata,
			outshape, std::vector<double>{0, 3, 1, 1});
		auto r = eigen::slice(edge);

		double* raw = r->get_ptr();
		r->assign();

		std::vector<double> expect_raw = {5, 6, 7};
		std::vector<double> got_raw(raw, raw + outshape.n_elems());
		EXPECT_ARREQ(expect_raw, got_raw);
	}
}


TEST(OPERATOR, GroupConcat)
{
	teq::Shape outshape({2, 4});
	MockEdge<double> edgea(
		teq::TensptrT(new MockTensor(teq::Shape({1, 4}))),
		std::vector<double>{2, 8, 4, 5}, outshape,
		std::vector<double>{0});
	MockEdge<double> edgeb(
		teq::TensptrT(new MockTensor(teq::Shape({1, 4}))),
		std::vector<double>{1, 0, 3, 9}, outshape);
	auto r = eigen::group_concat<double>({edgea, edgeb});

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {2, 1, 8, 0, 4, 3, 5, 9};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, GroupSum)
{
	teq::Shape outshape({2, 3});
	MockEdge<double> edgea(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
	MockEdge<double> edgeb(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{1, 0, 3, 9, 10, 11}, outshape);
	MockEdge<double> edgec(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{4.2, 1, 7.1, 1, 2, 1.1}, outshape);
	auto r = eigen::group_sum<double>({edgea, edgeb, edgec});

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {7.2, 9, 14.1, 15, 18, 19.1};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, GroupProd)
{
	teq::Shape outshape({2, 3});
	MockEdge<double> edgea(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7}, outshape);
	MockEdge<double> edgeb(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{1, 0, 3, 9, 10, 11}, outshape);
	MockEdge<double> edgec(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{4, 1, 7, 1, 2, 1}, outshape);
	auto r = eigen::group_prod<double>({edgea, edgeb, edgec});

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {8, 0, 84, 45, 120, 77};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Pad)
{
	teq::Shape outshape({4, 3});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape, std::vector<double>{1, 1});
	auto r = eigen::pad(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		0, 2, 8, 0,
		0, 4, 5, 0,
		0, 6, 7, 0,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Stride)
{
	teq::Shape outshape({2, 2});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape, std::vector<double>{1, 2});
	auto r = eigen::stride(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 6, 7,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Scatter)
{
	teq::Shape outshape({3, 3});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(teq::Shape({2, 2}))),
		std::vector<double>{2, 8, 4, 5},
		outshape, std::vector<double>{2, 2});
	auto r = eigen::scatter(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 0, 8,
		0, 0, 0,
		4, 0, 5,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Reverse)
{
	teq::Shape outshape({2, 3});
	MockEdge<double> edge(
		teq::TensptrT(new MockTensor(outshape)),
		std::vector<double>{2, 8, 4, 5, 6, 7},
		outshape, std::vector<double>{1});
	auto r = eigen::reverse(edge);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		6, 7, 4, 5, 2, 8
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


TEST(OPERATOR, Concat)
{
	teq::Shape outshape({3, 3});
	MockEdge<double> edgea(
		teq::TensptrT(new MockTensor(teq::Shape({2, 3}))),
		std::vector<double>{2, 8, 4, 5, 7, 6}, outshape,
		std::vector<double>{0});
	MockEdge<double> edgeb(
		teq::TensptrT(new MockTensor(teq::Shape({1, 3}))),
		std::vector<double>{1, 0, 3}, outshape);
	auto r = eigen::concat(edgea, edgeb);

	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2, 8, 1,
		4, 5, 0,
		7, 6, 3,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	EXPECT_ARREQ(expect_raw, got_raw);
}


#define _DBLARRCHECK(ARR, ARR2, GBOOL) { std::stringstream arrs, arrs2;\
	fmts::to_stream(arrs, ARR.begin(), ARR.end());\
	fmts::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	GBOOL(std::equal(ARR.begin(), ARR.end(), ARR2.begin())) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead"; }
#define ASSERT_ARRDBLEQ(ARR, ARR2) { std::stringstream arrs, arrs2;\
	fmts::to_stream(arrs, ARR.begin(), ARR.end());\
	fmts::to_stream(arrs2, ARR2.begin(), ARR2.end());\
	ASSERT_EQ(ARR.size(), ARR2.size()) <<\
		"expect list " << arrs.str() << ", got " << arrs2.str() << " instead";\
	for (size_t i = 0, n = ARR.size(); i < n; ++i){\
		ASSERT_DOUBLE_EQ(ARR[i], ARR2[i]) <<\
			"expect list " << arrs.str() << ", got " << arrs2.str() << " instead";\
	} }


static void elementary_unary (
	std::function<eigen::EigenptrT<double>(
		const eigen::iEigenEdge<double>&)> f,
	std::function<double(double)> unary,
	std::vector<double> invec = {-2, 8, -4, -5, 7, 6})
{
	std::vector<double> expect;
	std::transform(invec.begin(), invec.end(), std::back_inserter(expect),
		[&](double e) { return unary(e); });
	{
		teq::Shape shape({2, 3});
		MockEdge<double> edge(
			teq::TensptrT(new MockTensor(shape)), invec, shape);
		auto r = f(edge);

		double* raw = r->get_ptr();
		r->assign();

		std::vector<double> got_raw(raw, raw + shape.n_elems());
		ASSERT_ARRDBLEQ(expect, got_raw);
	}
	{
		teq::Shape shape({2, 1, 3});
		MockEdge<double> edge(
			teq::TensptrT(new MockTensor(shape)), invec, shape);
		auto r = f(edge);

		double* raw = r->get_ptr();
		r->assign();

		std::vector<double> got_raw(raw, raw + shape.n_elems());
		ASSERT_ARRDBLEQ(expect, got_raw);
	}
}


TEST(OPERATOR, Abs)
{
	elementary_unary(eigen::abs<double>, [](double e) { return std::abs(e); });
}


TEST(OPERATOR, Neg)
{
	elementary_unary(eigen::neg<double>, [](double e) { return -e; });
}


TEST(OPERATOR, Sin)
{
	elementary_unary(eigen::sin<double>, [](double e) { return std::sin(e); });
}


TEST(OPERATOR, Cos)
{
	elementary_unary(eigen::cos<double>, [](double e) { return std::cos(e); });
}


TEST(OPERATOR, Tan)
{
	elementary_unary(eigen::tan<double>, [](double e) { return std::tan(e); });
}


TEST(OPERATOR, Exp)
{
	elementary_unary(eigen::exp<double>, [](double e) { return std::exp(e); });
}


TEST(OPERATOR, Log)
{
	elementary_unary(eigen::log<double>, [](double e) { return std::log(e); },
		{3, 8, 2, 5, 7, 3});
}


TEST(OPERATOR, Sqrt)
{
	elementary_unary(eigen::sqrt<double>, [](double e) { return std::sqrt(e); },
		{3, 8, 2, 5, 7, 3});
}


TEST(OPERATOR, Round)
{
	elementary_unary(eigen::round<double>, [](double e) { return std::round(e); },
		{3.22, 8.51, 2.499, 5.2, 7.17, 3.79});
}


TEST(OPERATOR, Sigmoid)
{
	elementary_unary(eigen::sigmoid<double>,
		[](double e) { return 1. / (1. + std::exp(-e)); });
}


TEST(OPERATOR, Tanh)
{
	elementary_unary(eigen::tanh<double>,
		[](double e) { return std::tanh(e); });
}


TEST(OPERATOR, Square)
{
	elementary_unary(eigen::square<double>,
		[](double e) { return e * e; });
}


TEST(OPERATOR, Cube)
{
	elementary_unary(eigen::cube<double>,
		[](double e) { return e * e * e; });
}


TEST(OPERATOR, Convolution)
{
	teq::Shape outshape({3, 2});
	MockEdge<double> image(
		teq::TensptrT(new MockTensor(teq::Shape({3, 3}))),
		std::vector<double>{
			2, 8, 4,
			5, 7, 6,
			9, 1, 0,
		},
		outshape);
	MockEdge<double> kernel(
		teq::TensptrT(new MockTensor(teq::Shape({2}))),
		std::vector<double>{0.3, 0.6}, outshape,
		std::vector<double>{1});
	auto r = eigen::convolution(image, kernel);
	double* raw = r->get_ptr();
	r->assign();

	std::vector<double> expect_raw = {
		2 * 0.3 + 5 * 0.6, 8 * 0.3 + 7 * 0.6, 4 * 0.3 + 6 * 0.6,
		5 * 0.3 + 9 * 0.6, 7 * 0.3 + 1 * 0.6, 6 * 0.3 + 0 * 0.6,
	};
	std::vector<double> got_raw(raw, raw + outshape.n_elems());
	ASSERT_ARRDBLEQ(expect_raw, got_raw);
}


#endif // DISABLE_OPERATOR_TEST
