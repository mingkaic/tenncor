
#ifndef DISABLE_GLOBAL_RANDOM_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"


TEST(RANDOM, SetGet)
{
	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();

	auto generator = global::get_generator(ctx);
	void* origptr = generator.get();
	auto ogenerator = std::make_shared<global::Randomizer>();
	global::set_generator(ogenerator, ctx);
	EXPECT_NE(origptr, global::get_generator(ctx).get());
	EXPECT_EQ(ogenerator.get(), global::get_generator(ctx).get());
}


TEST(RANDOM, RandomizerStr)
{
	global::Randomizer rand;

	auto str = rand.get_str();
	auto sgen = rand.get_strgen();
	EXPECT_EQ(36, str.size());
	EXPECT_EQ(36, sgen().size());
}


TEST(RANDOM, RandomizerUniform)
{
	global::Randomizer rand;
	global::seed(0);

	auto inum = rand.unif_int(2, 7);
	auto igen = rand.unif_intgen(2, 7);
	EXPECT_LE(2, inum);
	EXPECT_GE(7, inum);
	auto ires = igen();
	EXPECT_LE(2, ires);
	EXPECT_GE(7, ires);

	auto fnum = rand.unif_dec(2, 7);
	auto fgen = rand.unif_decgen(2, 7);
	EXPECT_LE(2, fnum);
	EXPECT_GE(7, fnum);
	auto fres = fgen();
	EXPECT_LE(2, fres);
	EXPECT_GE(7, fres);
}


struct NormChecker
{
	NormChecker (double mean, double stdev) :
		mean_(mean), stdev_(stdev) {}

	void operator() (double d)
	{
		if ((mean_ - stdev_) < d && d < (mean_ + stdev_))
		{
			++stdev1_;
		}
		else if ((mean_ - 2 * stdev_) < d && d < (mean_ + 2 * stdev_))
		{
			++stdev2_;
		}
		else if ((mean_ - 3 * stdev_) < d && d < (mean_ + 3 * stdev_))
		{
			++stdev3_;
		}
		++count_;
	}

	void check (void)
	{
		auto prob68 = 100 * double(stdev1_) / (double) count_;
		auto prob95 = 100 * double(stdev1_ + stdev2_) / (double) count_;
		auto prob99 = 100 * double(stdev1_ + stdev2_ + stdev3_) / (double) count_;
		EXPECT_LE(63, prob68);
		EXPECT_GE(73, prob68); // 68 +/- 5
		EXPECT_LE(92, prob95);
		EXPECT_GE(98, prob95); // 95 +/- 3
		EXPECT_LE(96, prob99); // 99 +/- 3
	}

	double mean_;
	double stdev_;
	size_t stdev1_ = 0;
	size_t stdev2_ = 0;
	size_t stdev3_ = 0;
	size_t count_ = 0;
};


TEST(RANDOM, RandomizerNorm)
{
	global::Randomizer rand;
	global::seed(0);

	double mean = 2;
	double stdev = 3;
	NormChecker checker(mean, stdev);
	NormChecker checker2(mean, stdev);
	auto ngen = rand.norm_decgen(mean, stdev);

	for (size_t i = 0; i < 1000; ++i)
	{
		auto nnum = rand.norm_dec(mean, stdev);
		auto genres = ngen();
		checker(nnum);
		checker2(genres);
	}
	checker.check();
	checker2.check();
}


#endif // DISABLE_GLOBAL_RANDOM_TEST
