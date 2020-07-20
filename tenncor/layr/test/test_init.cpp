
#ifndef DISABLE_INIT_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "generated/api.hpp"


TEST(INIT, Zero)
{
	std::vector<teq::DimT> slist = {18, 9};
	std::string label = "abc";

	auto z = tenncor<double>().layer.zero_init()(teq::Shape(slist), label);
	auto shape = z->shape();
	ASSERT_ARREQ(slist, shape);

	double* d = (double*) z->device().data();
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		EXPECT_EQ(0, d[i]);
	}

	EXPECT_STREQ(z->to_string().c_str(), label.c_str());
}


TEST(INIT, VarianceScaling)
{
	std::vector<teq::DimT> slist = {18, 9, 3};
	std::string label = "def";
	double factor = 0.425;

	auto v1 = tenncor<double>().layer.variance_scaling_init(factor)(
		teq::Shape(slist), label);
	auto v2 = tenncor<double>().layer.variance_scaling_init(factor,
		[](teq::Shape s){ return s.at(2); })(teq::Shape(slist), label);
	{
		auto shape = v1->shape();
		ASSERT_ARREQ(slist, shape);

		double ex_stdev = std::sqrt(factor /
			((shape.at(0) + shape.at(1)) / 2));
		double upper = ex_stdev * 2;
		double lower = -upper;

		double* d = (double*) v1->device().data();
		for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
		{
			EXPECT_GT(upper, d[i]);
			EXPECT_LT(lower, d[i]);
		}

		EXPECT_STREQ(v1->to_string().c_str(), label.c_str());
	}
	{
		auto shape = v2->shape();
		ASSERT_ARREQ(slist, shape);

		double ex_stdev = std::sqrt(factor / shape.at(2));
		double bound = ex_stdev * 2;

		double* d = (double*) v2->device().data();
		for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
		{
			EXPECT_GT(bound, d[i]);
			EXPECT_LT(-bound, d[i]);
		}

		EXPECT_STREQ(v2->to_string().c_str(), label.c_str());
	}
}


TEST(INIT, UniformXavier)
{
	std::vector<teq::DimT> slist = {18, 9, 3};
	std::string label = "ghi";
	double factor = 0.712;

	auto x = tenncor<double>().layer.unif_xavier_init(factor)(
		teq::Shape(slist), label);

	auto shape = x->shape();
	ASSERT_ARREQ(slist, shape);

	double bound = factor * std::sqrt(6. / (shape.at(0) + shape.at(1)));

	double* d = (double*) x->device().data();
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		EXPECT_GT(bound, d[i]);
		EXPECT_LT(-bound, d[i]);
	}

	EXPECT_STREQ(x->to_string().c_str(), label.c_str());
}


// TEST(INIT, NormalXavier)
// {
// 	std::vector<teq::DimT> slist = {11, 12, 10};
// 	std::string label = "jkl";
// 	double factor = 0.172;

// 	auto x = tenncor<double>().layer.norm_xavier_init(factor)(
// 		teq::Shape(slist), label);

// 	auto shape = x->shape();
// 	ASSERT_ARREQ(slist, shape);

// 	double stdev = factor * std::sqrt(2. / (shape.at(0) + shape.at(1)));

// 	double* d = x->device().data();
// 	size_t stdevs[3];
// 	size_t n = shape.n_elems();
// 	for (size_t i = 0; i < n; ++i)
// 	{
// 		double c = d[i];
// 		if (-stdev < c && c < stdev)
// 		{
// 			++stdevs[0];
// 		}
// 		if (-2 * stdev < c && c < 2 * stdev)
// 		{
// 			++stdevs[1];
// 		}
// 		if (-3 * stdev < c && c < 3 * stdev)
// 		{
// 			++stdevs[2];
// 		}
// 	}
// 	double want_68 = (double) stdevs[0] / n;
// 	double want_95 = (double) stdevs[1] / n;
// 	double want_99 = (double) stdevs[2] / n;
// 	EXPECT_LT(60, want_68);
// 	EXPECT_GT(75, want_68);
// 	EXPECT_LT(90, want_95);
// 	EXPECT_GT(100, want_95);
// 	EXPECT_LT(95, want_99);
// 	EXPECT_GT(100, want_99);

// 	EXPECT_STREQ(x->to_string().c_str(), label.c_str());
// }


#endif // DISABLE_INIT_TEST
