
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

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(3, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, 3),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(4, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
	});

	EXPECT_DOUBLE_EQ(1, r3.lower_);
	EXPECT_DOUBLE_EQ(2, r3.upper_);
}


TEST(STABILIZER, Neg)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::NEG});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 3),
	});

	EXPECT_DOUBLE_EQ(-3, r1.lower_);
	EXPECT_DOUBLE_EQ(2, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_DOUBLE_EQ(-2, r2.lower_);
	EXPECT_DOUBLE_EQ(-1, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
	});

	EXPECT_DOUBLE_EQ(1, r3.lower_);
	EXPECT_DOUBLE_EQ(2, r3.upper_);
}


TEST(STABILIZER, Sin)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::SIN});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, 3), // > 2 pi
	});

	EXPECT_DOUBLE_EQ(-1, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_DOUBLE_EQ(std::sin(1), r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 6),
	});

	EXPECT_DOUBLE_EQ(-1, r3.lower_);
	EXPECT_DOUBLE_EQ(1, r3.upper_);

	auto r4 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 4),
	});

	EXPECT_DOUBLE_EQ(std::sin(4), r4.lower_);
	EXPECT_DOUBLE_EQ(std::sin(2), r4.upper_);
}


TEST(STABILIZER, Cos)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::COS});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, 3), // > 2 pi
	});

	EXPECT_DOUBLE_EQ(-1, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_DOUBLE_EQ(std::cos(2), r2.lower_);
	EXPECT_DOUBLE_EQ(std::cos(1), r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 7),
	});

	EXPECT_DOUBLE_EQ(-1, r3.lower_);
	EXPECT_DOUBLE_EQ(1, r3.upper_);

	auto r4 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 4),
	});

	EXPECT_DOUBLE_EQ(-1, r4.lower_);
	EXPECT_DOUBLE_EQ(std::cos(2), r4.upper_);
}


TEST(STABILIZER, Tan)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::TAN});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
	});

	double inf = std::numeric_limits<double>::infinity();
	EXPECT_EQ(-inf, r1.lower_);
	EXPECT_EQ(inf, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_EQ(-inf, r2.lower_);
	EXPECT_EQ(inf, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-0.5, 1),
	});

	EXPECT_DOUBLE_EQ(std::tan(-0.5), r3.lower_);
	EXPECT_DOUBLE_EQ(std::tan(1), r3.upper_);
}


TEST(STABILIZER, Exp)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::EXP});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_DOUBLE_EQ(std::exp(-2), r1.lower_);
	EXPECT_DOUBLE_EQ(std::exp(2), r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(
			std::log(std::numeric_limits<double>::min())*2, -2),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(std::exp(-2), r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(
			2, std::log(std::numeric_limits<double>::max())*2),
	});

	EXPECT_DOUBLE_EQ(std::exp(2), r3.lower_);
	EXPECT_EQ(std::numeric_limits<double>::infinity(), r3.upper_);
}


TEST(STABILIZER, Log)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::LOG});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_TRUE(std::isnan(r1.lower_));
	EXPECT_TRUE(std::isnan(r1.upper_));

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(0, 4),
	});

	EXPECT_EQ(-std::numeric_limits<double>::infinity(), r2.lower_);
	EXPECT_DOUBLE_EQ(std::log(4), r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 6),
	});

	EXPECT_DOUBLE_EQ(std::log(2), r3.lower_);
	EXPECT_DOUBLE_EQ(std::log(6), r3.upper_);
}


TEST(STABILIZER, Sqrt)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::SQRT});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_TRUE(std::isnan(r1.lower_));
	EXPECT_TRUE(std::isnan(r1.upper_));

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(0, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(2, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 6),
	});

	EXPECT_DOUBLE_EQ(std::sqrt(2), r3.lower_);
	EXPECT_DOUBLE_EQ(std::sqrt(6), r3.upper_);
}


TEST(STABILIZER, Round)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::ROUND});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2.2, 2.6),
	});

	EXPECT_DOUBLE_EQ(-2, r1.lower_);
	EXPECT_DOUBLE_EQ(3, r1.upper_);
}


TEST(STABILIZER, Square)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::SQUARE});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 3),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(9, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 3),
	});

	EXPECT_DOUBLE_EQ(4, r2.lower_);
	EXPECT_DOUBLE_EQ(9, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-3, -2),
	});

	EXPECT_DOUBLE_EQ(4, r3.lower_);
	EXPECT_DOUBLE_EQ(9, r3.upper_);
}


#endif // DISABLE_STABILIZER_TEST
