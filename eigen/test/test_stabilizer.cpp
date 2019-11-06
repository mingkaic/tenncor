
#ifndef DISABLE_STABILIZER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/opfunc.hpp"

#include "eigen/stabilizer.hpp"


TEST(STABILIZER, Abs)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::ABS});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-1, 3),
	});

	EXPECT_EQ(0, r1.lower_);
	EXPECT_EQ(3, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, 3),
	});

	EXPECT_EQ(0, r2.lower_);
	EXPECT_EQ(4, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
	});

	EXPECT_EQ(1, r3.lower_);
	EXPECT_EQ(2, r3.upper_);
}


#endif // DISABLE_STABILIZER_TEST
