
#ifndef DISABLE_STABILIZER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/stabilizer.hpp"


TEST(STABILIZER, Abs)
{
	teq::Shape shape({2, 3, 4});
	eteq::NodeptrT<double> a = eteq::make_constant_scalar<double>(0, shape);
	auto result = tenncor::abs(a);
	auto f = static_cast<teq::iFunctor*>(result->get_tensor().get());

	auto r1 = eteq::generate_range<double>(f, {
		estd::NumRange<double>(-1, 3),
	});

	EXPECT_EQ(0, r1.lower_);
	EXPECT_EQ(3, r1.upper_);

	auto r2 = eteq::generate_range<double>(f, {
		estd::NumRange<double>(-4, 3),
	});

	EXPECT_EQ(0, r2.lower_);
	EXPECT_EQ(4, r2.upper_);

	auto r3 = eteq::generate_range<double>(f, {
		estd::NumRange<double>(-2, -1),
	});

	EXPECT_EQ(1, r3.lower_);
	EXPECT_EQ(2, r3.upper_);
}


#endif // DISABLE_STABILIZER_TEST
