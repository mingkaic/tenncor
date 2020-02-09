
#ifndef DISABLE_STATS_TEST


#include "gtest/gtest.h"

#include "eteq/make.hpp"

#include "query/stats.hpp"


TEST(STATS, Compare)
{
	auto smol = eteq::make_constant_scalar<double>(1, teq::Shape({5, 1}));
	auto med = eteq::make_constant_scalar<double>(2, teq::Shape({5, 1}));
	auto big = eteq::make_constant_scalar<double>(2, teq::Shape({5, 2}));
	query::Stats a(smol.get(), 1);
	query::Stats b(med.get(), 1);
	query::Stats c(big.get(), 1);
	EXPECT_LT(a, b);
	EXPECT_LT(b, c);
}


#endif // DISABLE_STATS_TEST
