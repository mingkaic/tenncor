
#ifndef DISABLE_RANDOM_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eigen/random.hpp"


TEST(RANDOM, UniformValueDouble)
{
	double a = 4;
	double b = 16;
	double c = eigen::unif(a, b);
	EXPECT_LE(a, c);
	EXPECT_GE(b, c);
}


TEST(RANDOM, UniformValueInt)
{
	size_t a = 4;
	size_t b = 16;
	size_t c = eigen::unif(a, b);
	EXPECT_LE(a, c);
	EXPECT_GE(b, c);
}


TEST(RANDOM, UniformGenDouble)
{
	double a = 4;
	double b = 16;
	auto gen = eigen::unif_gen(a, b);
	std::vector<double> out(10);
	std::generate(out.begin(), out.end(), gen);
	for (auto c : out)
	{
		EXPECT_LE(a, c);
		EXPECT_GE(b, c);
	}
}


TEST(RANDOM, UniformGenInt)
{
	size_t a = 4;
	size_t b = 16;
	auto gen = eigen::unif_gen(a, b);
	std::vector<size_t> out(10);
	std::generate(out.begin(), out.end(), gen);
	for (auto c : out)
	{
		EXPECT_LE(a, c);
		EXPECT_GE(b, c);
	}
}


// TEST(RANDOM, NormalGen)
// {
// 	double a = 16;
// 	double b = 4;
// 	auto gen = eigen::norm_gen(a, b);
// 	std::vector<size_t> out(1000);
// 	std::generate(out.begin(), out.end(), gen);
// 	size_t stdevs[3];
// 	for (auto c : out)
// 	{
// 		if (a - b < c && c < a + b)
// 		{
// 			++stdevs[0];
// 		}
// 		if (a - 2 * b < c && c < a + 2 * b)
// 		{
// 			++stdevs[1];
// 		}
// 		if (a - 3 * b < c && c < a + 3 * b)
// 		{
// 			++stdevs[2];
// 		}
// 	}
// 	double want_68 = (double) stdevs[0] / 1000.;
// 	double want_95 = (double) stdevs[1] / 1000.;
// 	double want_99 = (double) stdevs[2] / 1000.;
// 	EXPECT_LT(60, want_68);
// 	EXPECT_GT(75, want_68);
// 	EXPECT_LT(90, want_95);
// 	EXPECT_GT(100, want_95);
// 	EXPECT_LT(95, want_99);
// 	EXPECT_GT(100, want_99);
// }


#endif // DISABLE_RANDOM_TEST
