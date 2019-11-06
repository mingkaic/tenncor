
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


TEST(STABILIZER, Neg)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::NEG});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 3),
	});

	EXPECT_EQ(-3, r1.lower_);
	EXPECT_EQ(2, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_EQ(-2, r2.lower_);
	EXPECT_EQ(-1, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
	});

	EXPECT_EQ(1, r3.lower_);
	EXPECT_EQ(2, r3.upper_);
}


TEST(STABILIZER, Sin)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::SIN});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, 3), // > 2 pi
	});

	EXPECT_EQ(-1, r1.lower_);
	EXPECT_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_EQ(std::sin(1), r2.lower_);
	EXPECT_EQ(1, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 6),
	});

	EXPECT_EQ(-1, r3.lower_);
	EXPECT_EQ(1, r3.upper_);

	auto r4 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 4),
	});

	EXPECT_EQ(std::sin(4), r4.lower_);
	EXPECT_EQ(std::sin(2), r4.upper_);
}


TEST(STABILIZER, Cos)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::COS});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, 3), // > 2 pi
	});

	EXPECT_EQ(-1, r1.lower_);
	EXPECT_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_EQ(std::cos(2), r2.lower_);
	EXPECT_EQ(std::cos(1), r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 7),
	});

	EXPECT_EQ(-1, r3.lower_);
	EXPECT_EQ(1, r3.upper_);

	auto r4 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 4),
	});

	EXPECT_EQ(-1, r4.lower_);
	EXPECT_EQ(std::cos(2), r4.upper_);
}


#endif // DISABLE_STABILIZER_TEST
