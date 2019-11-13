
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


TEST(STABILIZER, Cube)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::CUBE});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 3),
	});

	EXPECT_DOUBLE_EQ(-8, r1.lower_);
	EXPECT_DOUBLE_EQ(27, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 3),
	});

	EXPECT_DOUBLE_EQ(8, r2.lower_);
	EXPECT_DOUBLE_EQ(27, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-3, -2),
	});

	EXPECT_DOUBLE_EQ(-27, r3.lower_);
	EXPECT_DOUBLE_EQ(-8, r3.upper_);
}


TEST(STABILIZER, Sigmoid)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::SIGMOID});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-1000, 1000),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_DOUBLE_EQ(1 / (1 + std::exp(2)), r2.lower_);
	EXPECT_DOUBLE_EQ(1 / (1 + std::exp(-2)), r2.upper_);
}


TEST(STABILIZER, Tanh)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));
	auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::TANH});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-1000, 1000),
	});

	EXPECT_DOUBLE_EQ(-1, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_DOUBLE_EQ(std::tanh(-2), r2.lower_);
	EXPECT_DOUBLE_EQ(std::tanh(2), r2.upper_);
}


TEST(STABILIZER, Same)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	std::vector<egen::_GENERATED_OPCODE> ops = {
		egen::PERMUTE, egen::EXTEND,
		egen::RESHAPE, egen::REVERSE,
		egen::RAND_UNIF, egen::SLICE,
		egen::REDUCE_MAX, egen::REDUCE_MIN,
		egen::STRIDE,
	};
	for (egen::_GENERATED_OPCODE op : ops)
	{
		auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", op});

		auto r1 = eigen::generate_range<double>(f.get(), {
			estd::NumRange<double>(-245, 321),
		});

		EXPECT_DOUBLE_EQ(-245, r1.lower_);
		EXPECT_DOUBLE_EQ(321, r1.upper_);

		auto r2 = eigen::generate_range<double>(f.get(), {
			estd::NumRange<double>(-2.1, 2.5),
		});

		EXPECT_DOUBLE_EQ(-2.1, r2.lower_);
		EXPECT_DOUBLE_EQ(2.5, r2.upper_);
	}
}


TEST(STABILIZER, WithZeros)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	std::vector<egen::_GENERATED_OPCODE> ops = {
		egen::PAD, egen::SCATTER,
	};
	for (egen::_GENERATED_OPCODE op : ops)
	{
		auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"", op});

		auto r1 = eigen::generate_range<double>(f.get(), {
			estd::NumRange<double>(-3, 4),
		});

		EXPECT_DOUBLE_EQ(-3, r1.lower_);
		EXPECT_DOUBLE_EQ(4, r1.upper_);

		auto r2 = eigen::generate_range<double>(f.get(), {
			estd::NumRange<double>(-4, -2),
		});

		EXPECT_DOUBLE_EQ(-4, r2.lower_);
		EXPECT_DOUBLE_EQ(0, r2.upper_);

		auto r3 = eigen::generate_range<double>(f.get(), {
			estd::NumRange<double>(3, 5),
		});

		EXPECT_DOUBLE_EQ(0, r3.lower_);
		EXPECT_DOUBLE_EQ(5, r3.upper_);
	}
}


TEST(STABILIZER, Argmax)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f1 = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::ARGMAX},
		std::vector<double>{1});

	auto r1 = eigen::generate_range<double>(f1.get(), {
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(2, r1.upper_);

	auto f2 = std::make_shared<MockOpfunc>(a, teq::Opcode{"", egen::ARGMAX},
		std::vector<double>{8});

	auto r2 = eigen::generate_range<double>(f2.get(), {
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(shape.n_elems() - 1, r2.upper_);
}


TEST(STABILIZER, Select)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(MockEdgesT{
			MockEdge(a),
			MockEdge(a),
			MockEdge(a),
		}, teq::Opcode{"", egen::SELECT});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(4, r1.lower_);
	EXPECT_DOUBLE_EQ(6, r1.upper_);
}


TEST(STABILIZER, Concat)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a,
		teq::Opcode{"", egen::CONCAT});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(3, r1.lower_);
	EXPECT_DOUBLE_EQ(5, r1.upper_);
}


TEST(STABILIZER, GroupConcat)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(MockEdgesT{
			MockEdge(a),
			MockEdge(a),
			MockEdge(a),
		}, teq::Opcode{"", egen::GROUP_CONCAT});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(3, r1.lower_);
	EXPECT_DOUBLE_EQ(6, r1.upper_);
}


TEST(STABILIZER, Pow)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a,
		teq::Opcode{"", egen::POW});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(16, r1.lower_);
	EXPECT_DOUBLE_EQ(32, r1.upper_);

	// does not supporting complex yet
	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_TRUE(std::isnan(r2.lower_));
	EXPECT_TRUE(std::isnan(r2.upper_));

	// todo: test variable base
}


TEST(STABILIZER, Add)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a,
		teq::Opcode{"", egen::ADD});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(7, r1.lower_);
	EXPECT_DOUBLE_EQ(9, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(2, r2.lower_);
	EXPECT_DOUBLE_EQ(4, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(2, r3.lower_);
	EXPECT_DOUBLE_EQ(7, r3.upper_);
}


TEST(STABILIZER, GroupSum)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(MockEdgesT{
			MockEdge(a),
			MockEdge(a),
			MockEdge(a),
		}, teq::Opcode{"", egen::GROUP_SUM});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(12, r1.lower_);
	EXPECT_DOUBLE_EQ(15, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(-3, 2),
	});

	EXPECT_DOUBLE_EQ(-1, r2.lower_);
	EXPECT_DOUBLE_EQ(6, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
		estd::NumRange<double>(2, 4),
		estd::NumRange<double>(-5, 2),
	});

	EXPECT_DOUBLE_EQ(-5, r3.lower_);
	EXPECT_DOUBLE_EQ(8, r3.upper_);
}


TEST(STABILIZER, Sub)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a,
		teq::Opcode{"", egen::SUB});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-2, r1.lower_);
	EXPECT_DOUBLE_EQ(0, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(2, r2.upper_);
}


TEST(STABILIZER, Mul)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a,
		teq::Opcode{"", egen::MUL});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(12, r1.lower_);
	EXPECT_DOUBLE_EQ(20, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-10, r2.lower_);
	EXPECT_DOUBLE_EQ(-4, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-10, r3.lower_);
	EXPECT_DOUBLE_EQ(10, r3.upper_);
}


TEST(STABILIZER, GroupProd)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(MockEdgesT{
			MockEdge(a),
			MockEdge(a),
			MockEdge(a),
		}, teq::Opcode{"", egen::GROUP_PROD});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(60, r1.lower_);
	EXPECT_DOUBLE_EQ(120, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(-3, 2),
	});

	EXPECT_DOUBLE_EQ(-20, r2.lower_);
	EXPECT_DOUBLE_EQ(30, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
		estd::NumRange<double>(2, 4),
		estd::NumRange<double>(-5, 2),
	});

	EXPECT_DOUBLE_EQ(-40, r3.lower_);
	EXPECT_DOUBLE_EQ(40, r3.upper_);
}


TEST(STABILIZER, Div)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a,
		teq::Opcode{"", egen::DIV});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(3./5, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-2./4, r2.lower_);
	EXPECT_DOUBLE_EQ(-1./5, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-1./2, r3.lower_);
	EXPECT_DOUBLE_EQ(1./2, r3.upper_);
}


TEST(STABILIZER, Min)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a,
		teq::Opcode{"", egen::MIN});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(3, r1.lower_);
	EXPECT_DOUBLE_EQ(4, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 6),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-2, r2.lower_);
	EXPECT_DOUBLE_EQ(5, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
		estd::NumRange<double>(-2, 5),
	});

	EXPECT_DOUBLE_EQ(-2, r3.lower_);
	EXPECT_DOUBLE_EQ(2, r3.upper_);
}


TEST(STABILIZER, Max)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a,
		teq::Opcode{"", egen::MAX});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(4, r1.lower_);
	EXPECT_DOUBLE_EQ(5, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 6),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(4, r2.lower_);
	EXPECT_DOUBLE_EQ(6, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 2),
		estd::NumRange<double>(-2, 5),
	});

	EXPECT_DOUBLE_EQ(1, r3.lower_);
	EXPECT_DOUBLE_EQ(5, r3.upper_);
}


TEST(STABILIZER, Eq)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a, teq::Opcode{"", egen::EQ});

	// overlaps
	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, -2),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 5),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	// constant absolute false
	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(2, 2),
	});

	EXPECT_DOUBLE_EQ(0, r3.lower_);
	EXPECT_DOUBLE_EQ(0, r3.upper_);

	// constant absolute truth
	auto r4 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(1, r4.lower_);
	EXPECT_DOUBLE_EQ(1, r4.upper_);

	// ranged absolute false by having constant outbound
	auto r5 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(5, 5),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r5.lower_);
	EXPECT_DOUBLE_EQ(0, r5.upper_);

	auto r6 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-3, -3),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r6.lower_);
	EXPECT_DOUBLE_EQ(0, r6.upper_);
}


TEST(STABILIZER, Neq)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a, teq::Opcode{"", egen::NEQ});

	// overlaps
	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, -2),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 5),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	// constant absolute truth
	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(2, 2),
	});

	EXPECT_DOUBLE_EQ(1, r3.lower_);
	EXPECT_DOUBLE_EQ(1, r3.upper_);

	// constant absolute false
	auto r4 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(0, r4.lower_);
	EXPECT_DOUBLE_EQ(0, r4.upper_);

	// ranged absolute truth by having constant outbound
	auto r5 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(5, 5),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(1, r5.lower_);
	EXPECT_DOUBLE_EQ(1, r5.upper_);

	auto r6 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-3, -3),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(1, r6.lower_);
	EXPECT_DOUBLE_EQ(1, r6.upper_);
}


TEST(STABILIZER, Lt)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a, teq::Opcode{"", egen::LT});

	// overlaps
	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, -2),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 5),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	// constant absolute false
	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(2, 2),
	});

	EXPECT_DOUBLE_EQ(0, r3.lower_);
	EXPECT_DOUBLE_EQ(0, r3.upper_);

	// constant absolute truth
	auto r4 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 2),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(1, r4.lower_);
	EXPECT_DOUBLE_EQ(1, r4.upper_);

	// between uncertainty with constant
	auto r5 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 4),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(0, r5.lower_);
	EXPECT_DOUBLE_EQ(1, r5.upper_);

	auto r6 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r6.lower_);
	EXPECT_DOUBLE_EQ(1, r6.upper_);

	// ranged absolute false
	auto r7 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(5, 6),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r7.lower_);
	EXPECT_DOUBLE_EQ(0, r7.upper_);

	// ranged absolute truth
	auto r8 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 4),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(1, r8.lower_);
	EXPECT_DOUBLE_EQ(1, r8.upper_);
}


TEST(STABILIZER, Gt)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f = std::make_shared<MockOpfunc>(a, a, teq::Opcode{"", egen::GT});

	// overlaps
	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, -2),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 5),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	// constant absolute false
	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(2, 2),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(0, r3.lower_);
	EXPECT_DOUBLE_EQ(0, r3.upper_);

	// constant absolute truth
	auto r4 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(2, 2),
	});

	EXPECT_DOUBLE_EQ(1, r4.lower_);
	EXPECT_DOUBLE_EQ(1, r4.upper_);

	// between uncertainty with constant
	auto r5 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 4),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(0, r5.lower_);
	EXPECT_DOUBLE_EQ(1, r5.upper_);

	auto r6 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r6.lower_);
	EXPECT_DOUBLE_EQ(1, r6.upper_);

	// ranged absolute false
	auto r7 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-2, 4),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(0, r7.lower_);
	EXPECT_DOUBLE_EQ(0, r7.upper_);

	// ranged absolute truth
	auto r8 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(5, 6),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(1, r8.lower_);
	EXPECT_DOUBLE_EQ(1, r8.upper_);
}


TEST(STABILIZER, ReduceSum)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f1 = std::make_shared<MockOpfunc>(a,
		teq::Opcode{"", egen::REDUCE_SUM},
		std::vector<double>{1, 2, 8, 8, 8, 8, 8, 8});

	auto r1 = eigen::generate_range<double>(f1.get(), {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(-2 * 12, r1.lower_);
	EXPECT_DOUBLE_EQ(4 * 12, r1.upper_);

	auto f2 = std::make_shared<MockOpfunc>(a,
		teq::Opcode{"", egen::REDUCE_SUM},
		std::vector<double>{8, 8, 8, 8, 8, 8, 8, 8});

	auto r2 = eigen::generate_range<double>(f2.get(), {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ((-2.) * shape.n_elems(), r2.lower_);
	EXPECT_DOUBLE_EQ(4 * shape.n_elems(), r2.upper_);
}


TEST(STABILIZER, ReduceProd)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	auto f1 = std::make_shared<MockOpfunc>(a,
		teq::Opcode{"", egen::REDUCE_PROD},
		std::vector<double>{0, 2, 8, 8, 8, 8, 8, 8});

	auto r1 = eigen::generate_range<double>(f1.get(), {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(std::pow(4, 8), r1.upper_);

	auto r2 = eigen::generate_range<double>(f1.get(), {
		estd::NumRange<double>(-4, 2),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(std::pow(4, 8), r2.upper_);

	auto f2 = std::make_shared<MockOpfunc>(a,
		teq::Opcode{"", egen::REDUCE_PROD},
		std::vector<double>{8, 8, 8, 8, 8, 8, 8, 8});

	auto r3 = eigen::generate_range<double>(f2.get(), {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r3.lower_);
	EXPECT_DOUBLE_EQ(std::pow(4, shape.n_elems()), r3.upper_);

	auto r4 = eigen::generate_range<double>(f2.get(), {
		estd::NumRange<double>(-4, 2),
	});

	EXPECT_DOUBLE_EQ(0, r4.lower_);
	EXPECT_DOUBLE_EQ(std::pow(4, shape.n_elems()), r4.upper_);

	auto f3 = std::make_shared<MockOpfunc>(a,
		teq::Opcode{"", egen::REDUCE_PROD},
		std::vector<double>{1, 8, 8, 8, 8, 8, 8, 8});

	auto r5 = eigen::generate_range<double>(f3.get(), {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(-8, r5.lower_);
	EXPECT_DOUBLE_EQ(64, r5.upper_);
}


TEST(STABILIZER, Matmul)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	MockEdge fa(a, {}, {0, 2, 8, 8, 8, 8, 8, 8});

	auto f = std::make_shared<MockOpfunc>(MockEdgesT{fa, fa},
		teq::Opcode{"", egen::MATMUL});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, 2),
		estd::NumRange<double>(-3, 3),
	});

	EXPECT_DOUBLE_EQ(-12 * 8, r1.lower_);
	EXPECT_DOUBLE_EQ(12 * 8, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 3),
		estd::NumRange<double>(-3, 2),
	});

	EXPECT_DOUBLE_EQ(-9 * 8, r2.lower_);
	EXPECT_DOUBLE_EQ(6 * 8, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 3),
		estd::NumRange<double>(0.5, 2),
	});

	EXPECT_DOUBLE_EQ(0.5 * 8, r3.lower_);
	EXPECT_DOUBLE_EQ(6 * 8, r3.upper_);
}


TEST(STABILIZER, Conv)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT a(new MockTensor(shape));

	MockEdge fa(a, {}, {0, 2, 8, 8, 8, 8, 8, 8});

	auto f = std::make_shared<MockOpfunc>(MockEdgesT{fa, fa},
		teq::Opcode{"", egen::CONV});

	auto r1 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(-4, 2),
		estd::NumRange<double>(-3, 3),
	});

	EXPECT_DOUBLE_EQ(-12 * 6, r1.lower_);
	EXPECT_DOUBLE_EQ(12 * 6, r1.upper_);

	auto r2 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 3),
		estd::NumRange<double>(-3, 2),
	});

	EXPECT_DOUBLE_EQ(-9 * 6, r2.lower_);
	EXPECT_DOUBLE_EQ(6 * 6, r2.upper_);

	auto r3 = eigen::generate_range<double>(f.get(), {
		estd::NumRange<double>(1, 3),
		estd::NumRange<double>(0.5, 2),
	});

	EXPECT_DOUBLE_EQ(0.5 * 6, r3.lower_);
	EXPECT_DOUBLE_EQ(6 * 6, r3.upper_);
}


#endif // DISABLE_STABILIZER_TEST
