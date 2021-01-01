
#ifndef DISABLE_UTILS_STABILIZER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/teq/mock/mock.hpp"

#include "internal/utils/stabilizer/stabilizer.hpp"


using ::testing::Const;
using ::testing::Return;


TEST(STABILIZER, Abs)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::ABS, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-1, 3),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(3, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, 3),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(4, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, -1),
	});

	EXPECT_DOUBLE_EQ(1, r3.lower_);
	EXPECT_DOUBLE_EQ(2, r3.upper_);
}


TEST(STABILIZER, Neg)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::NEG, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 3),
	});

	EXPECT_DOUBLE_EQ(-3, r1.lower_);
	EXPECT_DOUBLE_EQ(2, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_DOUBLE_EQ(-2, r2.lower_);
	EXPECT_DOUBLE_EQ(-1, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, -1),
	});

	EXPECT_DOUBLE_EQ(1, r3.lower_);
	EXPECT_DOUBLE_EQ(2, r3.upper_);
}


TEST(STABILIZER, Sin)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::SIN, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, 3), // > 2 pi
	});

	EXPECT_DOUBLE_EQ(-1, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_DOUBLE_EQ(std::sin(1), r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 6),
	});

	EXPECT_DOUBLE_EQ(-1, r3.lower_);
	EXPECT_DOUBLE_EQ(1, r3.upper_);

	auto r4 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 4),
	});

	EXPECT_DOUBLE_EQ(std::sin(4), r4.lower_);
	EXPECT_DOUBLE_EQ(std::sin(2), r4.upper_);
}


TEST(STABILIZER, Cos)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::COS, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, 3), // > 2 pi
	});

	EXPECT_DOUBLE_EQ(-1, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_DOUBLE_EQ(std::cos(2), r2.lower_);
	EXPECT_DOUBLE_EQ(std::cos(1), r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 7),
	});

	EXPECT_DOUBLE_EQ(-1, r3.lower_);
	EXPECT_DOUBLE_EQ(1, r3.upper_);

	auto r4 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 4),
	});

	EXPECT_DOUBLE_EQ(-1, r4.lower_);
	EXPECT_DOUBLE_EQ(std::cos(2), r4.upper_);
}


TEST(STABILIZER, Tan)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::TAN, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
	});

	double inf = std::numeric_limits<double>::infinity();
	EXPECT_EQ(-inf, r1.lower_);
	EXPECT_EQ(inf, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 2),
	});

	EXPECT_EQ(-inf, r2.lower_);
	EXPECT_EQ(inf, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-0.5, 1),
	});

	EXPECT_DOUBLE_EQ(std::tan(-0.5), r3.lower_);
	EXPECT_DOUBLE_EQ(std::tan(1), r3.upper_);
}


TEST(STABILIZER, Exp)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::EXP, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_DOUBLE_EQ(std::exp(-2), r1.lower_);
	EXPECT_DOUBLE_EQ(std::exp(2), r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(
			std::log(std::numeric_limits<double>::min())*2, -2),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(std::exp(-2), r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(
			2, std::log(std::numeric_limits<double>::max())*2),
	});

	EXPECT_DOUBLE_EQ(std::exp(2), r3.lower_);
	EXPECT_EQ(std::numeric_limits<double>::infinity(), r3.upper_);
}


TEST(STABILIZER, Log)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::LOG, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_TRUE(std::isnan(r1.lower_));
	EXPECT_TRUE(std::isnan(r1.upper_));

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(0, 4),
	});

	EXPECT_EQ(-std::numeric_limits<double>::infinity(), r2.lower_);
	EXPECT_DOUBLE_EQ(std::log(4), r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 6),
	});

	EXPECT_DOUBLE_EQ(std::log(2), r3.lower_);
	EXPECT_DOUBLE_EQ(std::log(6), r3.upper_);
}


TEST(STABILIZER, Sqrt)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::SQRT, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_TRUE(std::isnan(r1.lower_));
	EXPECT_TRUE(std::isnan(r1.upper_));

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(0, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(2, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 6),
	});

	EXPECT_DOUBLE_EQ(std::sqrt(2), r3.lower_);
	EXPECT_DOUBLE_EQ(std::sqrt(6), r3.upper_);
}


TEST(STABILIZER, Round)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::ROUND, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2.2, 2.6),
	});

	EXPECT_DOUBLE_EQ(-2, r1.lower_);
	EXPECT_DOUBLE_EQ(3, r1.upper_);
}


TEST(STABILIZER, Square)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::SQUARE, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 3),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(9, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 3),
	});

	EXPECT_DOUBLE_EQ(4, r2.lower_);
	EXPECT_DOUBLE_EQ(9, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-3, -2),
	});

	EXPECT_DOUBLE_EQ(4, r3.lower_);
	EXPECT_DOUBLE_EQ(9, r3.upper_);
}


TEST(STABILIZER, Cube)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::CUBE, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 3),
	});

	EXPECT_DOUBLE_EQ(-8, r1.lower_);
	EXPECT_DOUBLE_EQ(27, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 3),
	});

	EXPECT_DOUBLE_EQ(8, r2.lower_);
	EXPECT_DOUBLE_EQ(27, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-3, -2),
	});

	EXPECT_DOUBLE_EQ(-27, r3.lower_);
	EXPECT_DOUBLE_EQ(-8, r3.upper_);
}


TEST(STABILIZER, Sigmoid)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::SIGMOID, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-1000, 1000),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_DOUBLE_EQ(1 / (1 + std::exp(2)), r2.lower_);
	EXPECT_DOUBLE_EQ(1 / (1 + std::exp(-2)), r2.upper_);
}


TEST(STABILIZER, Tanh)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);
	auto f = make_fnc("", egen::TANH, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-1000, 1000),
	});

	EXPECT_DOUBLE_EQ(-1, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
	});

	EXPECT_DOUBLE_EQ(std::tanh(-2), r2.lower_);
	EXPECT_DOUBLE_EQ(std::tanh(2), r2.upper_);
}


TEST(STABILIZER, Same)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	std::vector<egen::_GENERATED_OPCODE> ops = {
		egen::PERMUTE, egen::EXTEND,
		egen::RESHAPE, egen::REVERSE,
		egen::RAND_UNIF, egen::SLICE,
		egen::REDUCE_MAX, egen::REDUCE_MIN,
		egen::STRIDE,
	};
	for (egen::_GENERATED_OPCODE op : ops)
	{
		auto f = make_fnc("", op, teq::TensptrsT{a});
		EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

		auto r1 = eigen::generate_range<double>(*f, {
			estd::NumRange<double>(-245, 321),
		});

		EXPECT_DOUBLE_EQ(-245, r1.lower_);
		EXPECT_DOUBLE_EQ(321, r1.upper_);

		auto r2 = eigen::generate_range<double>(*f, {
			estd::NumRange<double>(-2.1, 2.5),
		});

		EXPECT_DOUBLE_EQ(-2.1, r2.lower_);
		EXPECT_DOUBLE_EQ(2.5, r2.upper_);
	}
}


TEST(STABILIZER, WithZeros)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	std::vector<egen::_GENERATED_OPCODE> ops = {
		egen::PAD, egen::SCATTER,
	};
	for (egen::_GENERATED_OPCODE op : ops)
	{
		auto f = make_fnc("", op, teq::TensptrsT{a});
		EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

		auto r1 = eigen::generate_range<double>(*f, {
			estd::NumRange<double>(-3, 4),
		});

		EXPECT_DOUBLE_EQ(-3, r1.lower_);
		EXPECT_DOUBLE_EQ(4, r1.upper_);

		auto r2 = eigen::generate_range<double>(*f, {
			estd::NumRange<double>(-4, -2),
		});

		EXPECT_DOUBLE_EQ(-4, r2.lower_);
		EXPECT_DOUBLE_EQ(0, r2.upper_);

		auto r3 = eigen::generate_range<double>(*f, {
			estd::NumRange<double>(3, 5),
		});

		EXPECT_DOUBLE_EQ(0, r3.lower_);
		EXPECT_DOUBLE_EQ(5, r3.upper_);
	}
}


TEST(STABILIZER, Argmax)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	marsh::Number<int64_t> numobj(1);
	auto f1 = make_fnc("", egen::ARGMAX, teq::TensptrsT{a});
	EXPECT_CALL(*f1, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f1, get_attr(eigen::Packer<teq::RankT>::key_)).WillRepeatedly(Return(&numobj));
	EXPECT_CALL(Const(*f1), get_attr(eigen::Packer<teq::RankT>::key_)).WillRepeatedly(Return(&numobj));

	auto r1 = eigen::generate_range<double>(*f1, {
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(2, r1.upper_);

	marsh::Number<int64_t> numobj2(8);
	auto f2 = make_fnc("", egen::ARGMAX, teq::TensptrsT{a});
	EXPECT_CALL(*f2, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*f2, get_attr(eigen::Packer<teq::RankT>::key_)).WillRepeatedly(Return(&numobj2));
	EXPECT_CALL(Const(*f2), get_attr(eigen::Packer<teq::RankT>::key_)).WillRepeatedly(Return(&numobj2));

	auto r2 = eigen::generate_range<double>(*f2, {
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(shape.n_elems() - 1, r2.upper_);
}


TEST(STABILIZER, Select)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::SELECT, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
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
	auto a = make_var(shape);

	auto f = make_fnc("", egen::CONCAT, teq::TensptrsT{a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(3, r1.lower_);
	EXPECT_DOUBLE_EQ(5, r1.upper_);
}


TEST(STABILIZER, GroupConcat)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::CONCAT, teq::TensptrsT{a, a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
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
	auto a = make_var(shape);

	auto f = make_fnc("", egen::POW, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(16, r1.lower_);
	EXPECT_DOUBLE_EQ(32, r1.upper_);

	// does not supporting complex yet
	auto r2 = eigen::generate_range<double>(*f, {
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
	auto a = make_var(shape);

	auto f = make_fnc("", egen::ADD, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(7, r1.lower_);
	EXPECT_DOUBLE_EQ(9, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(2, r2.lower_);
	EXPECT_DOUBLE_EQ(4, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(2, r3.lower_);
	EXPECT_DOUBLE_EQ(7, r3.upper_);
}


TEST(STABILIZER, GroupSum)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::ADD, teq::TensptrsT{a, a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(12, r1.lower_);
	EXPECT_DOUBLE_EQ(15, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(-3, 2),
	});

	EXPECT_DOUBLE_EQ(-1, r2.lower_);
	EXPECT_DOUBLE_EQ(6, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
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
	auto a = make_var(shape);

	auto f = make_fnc("", egen::SUB, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-2, r1.lower_);
	EXPECT_DOUBLE_EQ(0, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(2, r2.upper_);
}


TEST(STABILIZER, Mul)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::MUL, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(12, r1.lower_);
	EXPECT_DOUBLE_EQ(20, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-10, r2.lower_);
	EXPECT_DOUBLE_EQ(-4, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-10, r3.lower_);
	EXPECT_DOUBLE_EQ(10, r3.upper_);
}


TEST(STABILIZER, GroupProd)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::MUL, teq::TensptrsT{a, a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(60, r1.lower_);
	EXPECT_DOUBLE_EQ(120, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
		estd::NumRange<double>(-3, 2),
	});

	EXPECT_DOUBLE_EQ(-20, r2.lower_);
	EXPECT_DOUBLE_EQ(30, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
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
	auto a = make_var(shape);

	auto f = make_fnc("", egen::DIV, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(3./5, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, -1),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-2./4, r2.lower_);
	EXPECT_DOUBLE_EQ(-1./5, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 2),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-1./2, r3.lower_);
	EXPECT_DOUBLE_EQ(1./2, r3.upper_);
}


TEST(STABILIZER, Min)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::MIN, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(3, r1.lower_);
	EXPECT_DOUBLE_EQ(4, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 6),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(-2, r2.lower_);
	EXPECT_DOUBLE_EQ(5, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 2),
		estd::NumRange<double>(-2, 5),
	});

	EXPECT_DOUBLE_EQ(-2, r3.lower_);
	EXPECT_DOUBLE_EQ(2, r3.upper_);
}


TEST(STABILIZER, Max)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::MAX, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 4),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(4, r1.lower_);
	EXPECT_DOUBLE_EQ(5, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 6),
		estd::NumRange<double>(4, 5),
	});

	EXPECT_DOUBLE_EQ(4, r2.lower_);
	EXPECT_DOUBLE_EQ(6, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 2),
		estd::NumRange<double>(-2, 5),
	});

	EXPECT_DOUBLE_EQ(1, r3.lower_);
	EXPECT_DOUBLE_EQ(5, r3.upper_);
}


TEST(STABILIZER, Eq)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::EQ, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	// overlaps
	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, -2),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 5),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	// constant absolute false
	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(2, 2),
	});

	EXPECT_DOUBLE_EQ(0, r3.lower_);
	EXPECT_DOUBLE_EQ(0, r3.upper_);

	// constant absolute truth
	auto r4 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(1, r4.lower_);
	EXPECT_DOUBLE_EQ(1, r4.upper_);

	// ranged absolute false by having constant outbound
	auto r5 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(5, 5),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r5.lower_);
	EXPECT_DOUBLE_EQ(0, r5.upper_);

	auto r6 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-3, -3),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r6.lower_);
	EXPECT_DOUBLE_EQ(0, r6.upper_);
}


TEST(STABILIZER, Neq)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::NEQ, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	// overlaps
	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, -2),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 5),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	// constant absolute truth
	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(2, 2),
	});

	EXPECT_DOUBLE_EQ(1, r3.lower_);
	EXPECT_DOUBLE_EQ(1, r3.upper_);

	// constant absolute false
	auto r4 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(0, r4.lower_);
	EXPECT_DOUBLE_EQ(0, r4.upper_);

	// ranged absolute truth by having constant outbound
	auto r5 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(5, 5),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(1, r5.lower_);
	EXPECT_DOUBLE_EQ(1, r5.upper_);

	auto r6 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-3, -3),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(1, r6.lower_);
	EXPECT_DOUBLE_EQ(1, r6.upper_);
}


TEST(STABILIZER, Lt)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::LT, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	// overlaps
	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, -2),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 5),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	// constant absolute false
	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(2, 2),
	});

	EXPECT_DOUBLE_EQ(0, r3.lower_);
	EXPECT_DOUBLE_EQ(0, r3.upper_);

	// constant absolute truth
	auto r4 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 2),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(1, r4.lower_);
	EXPECT_DOUBLE_EQ(1, r4.upper_);

	// between uncertainty with constant
	auto r5 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 4),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(0, r5.lower_);
	EXPECT_DOUBLE_EQ(1, r5.upper_);

	auto r6 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r6.lower_);
	EXPECT_DOUBLE_EQ(1, r6.upper_);

	// ranged absolute false
	auto r7 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(5, 6),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r7.lower_);
	EXPECT_DOUBLE_EQ(0, r7.upper_);

	// ranged absolute truth
	auto r8 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 4),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(1, r8.lower_);
	EXPECT_DOUBLE_EQ(1, r8.upper_);
}


TEST(STABILIZER, Gt)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto f = make_fnc("", egen::GT, teq::TensptrsT{a, a});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	// overlaps
	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, -2),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(1, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 5),
		estd::NumRange<double>(-3, 4),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(1, r2.upper_);

	// constant absolute false
	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(2, 2),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(0, r3.lower_);
	EXPECT_DOUBLE_EQ(0, r3.upper_);

	// constant absolute truth
	auto r4 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(2, 2),
	});

	EXPECT_DOUBLE_EQ(1, r4.lower_);
	EXPECT_DOUBLE_EQ(1, r4.upper_);

	// between uncertainty with constant
	auto r5 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 4),
		estd::NumRange<double>(3, 3),
	});

	EXPECT_DOUBLE_EQ(0, r5.lower_);
	EXPECT_DOUBLE_EQ(1, r5.upper_);

	auto r6 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(3, 3),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r6.lower_);
	EXPECT_DOUBLE_EQ(1, r6.upper_);

	// ranged absolute false
	auto r7 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-2, 4),
		estd::NumRange<double>(5, 6),
	});

	EXPECT_DOUBLE_EQ(0, r7.lower_);
	EXPECT_DOUBLE_EQ(0, r7.upper_);

	// ranged absolute truth
	auto r8 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(5, 6),
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(1, r8.lower_);
	EXPECT_DOUBLE_EQ(1, r8.upper_);
}


TEST(STABILIZER, ReduceSum)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto setkey = eigen::Packer<std::set<teq::RankT>>::key_;
	marsh::NumArray<int64_t> numsobj(std::vector<int64_t>{1, 2});
	marsh::NumArray<int64_t> numsobj2(std::vector<int64_t>{0, 1, 2});
	marsh::NumArray<int64_t> numsobj3(std::vector<int64_t>{1});

	auto f1 = make_fnc("", egen::REDUCE_SUM, teq::TensptrsT{a});
	EXPECT_CALL(*f1, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*f1, get_attr(setkey)).WillRepeatedly(Return(&numsobj));
	EXPECT_CALL(Const(*f1), get_attr(setkey)).WillRepeatedly(Return(&numsobj));

	auto r1 = eigen::generate_range<double>(*f1, {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(-2 * 12, r1.lower_);
	EXPECT_DOUBLE_EQ(4 * 12, r1.upper_);

	EXPECT_CALL(*f1, get_attr(setkey)).WillRepeatedly(Return(&numsobj2));
	EXPECT_CALL(Const(*f1), get_attr(setkey)).WillRepeatedly(Return(&numsobj2));

	auto r2 = eigen::generate_range<double>(*f1, {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ((-2.) * shape.n_elems(), r2.lower_);
	EXPECT_DOUBLE_EQ(4 * shape.n_elems(), r2.upper_);
}


TEST(STABILIZER, ReduceProd)
{
	teq::Shape shape({2, 3, 4});
	auto a = make_var(shape);

	auto setkey = eigen::Packer<std::set<teq::RankT>>::key_;
	marsh::NumArray<int64_t> numsobj(std::vector<int64_t>{0, 2});
	marsh::NumArray<int64_t> numsobj2(std::vector<int64_t>{0, 1, 2});
	marsh::NumArray<int64_t> numsobj3(std::vector<int64_t>{1});

	auto f1 = make_fnc("", egen::REDUCE_PROD, teq::TensptrsT{a});
	EXPECT_CALL(*f1, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*f1, get_attr(setkey)).WillRepeatedly(Return(&numsobj));
	EXPECT_CALL(Const(*f1), get_attr(setkey)).WillRepeatedly(Return(&numsobj));

	auto r1 = eigen::generate_range<double>(*f1, {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r1.lower_);
	EXPECT_DOUBLE_EQ(std::pow(4, 8), r1.upper_);

	auto r2 = eigen::generate_range<double>(*f1, {
		estd::NumRange<double>(-4, 2),
	});

	EXPECT_DOUBLE_EQ(0, r2.lower_);
	EXPECT_DOUBLE_EQ(std::pow(4, 8), r2.upper_);

	EXPECT_CALL(*f1, get_attr(setkey)).WillRepeatedly(Return(&numsobj2));
	EXPECT_CALL(Const(*f1), get_attr(setkey)).WillRepeatedly(Return(&numsobj2));

	auto r3 = eigen::generate_range<double>(*f1, {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(0, r3.lower_);
	EXPECT_DOUBLE_EQ(std::pow(4, shape.n_elems()), r3.upper_);

	auto r4 = eigen::generate_range<double>(*f1, {
		estd::NumRange<double>(-4, 2),
	});

	EXPECT_DOUBLE_EQ(0, r4.lower_);
	EXPECT_DOUBLE_EQ(std::pow(4, shape.n_elems()), r4.upper_);

	EXPECT_CALL(*f1, get_attr(setkey)).WillRepeatedly(Return(&numsobj3));
	EXPECT_CALL(Const(*f1), get_attr(setkey)).WillRepeatedly(Return(&numsobj3));

	auto r5 = eigen::generate_range<double>(*f1, {
		estd::NumRange<double>(-2, 4),
	});

	EXPECT_DOUBLE_EQ(-8, r5.lower_);
	EXPECT_DOUBLE_EQ(64, r5.upper_);
}


TEST(STABILIZER, Matmul)
{
	teq::Shape shape({4, 3, 2});
	teq::Shape shape2({2, 3, 4});
	auto a = make_var(shape);
	auto b = make_var(shape2);

	auto setkey = eigen::Packer<eigen::PairVecT<teq::RankT>>::key_;
	marsh::NumArray<int64_t> numsobj(eigen::encode_pair(eigen::PairVecT<teq::RankT>{{0, 2}}));

	auto f = make_fnc("", egen::MATMUL, teq::TensptrsT{a, b});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*f, get_attr(setkey)).WillRepeatedly(Return(&numsobj));
	EXPECT_CALL(Const(*f), get_attr(setkey)).WillRepeatedly(Return(&numsobj));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, 2),
		estd::NumRange<double>(-3, 3),
	});

	EXPECT_DOUBLE_EQ(-12 * 4, r1.lower_);
	EXPECT_DOUBLE_EQ(12 * 4, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 3),
		estd::NumRange<double>(-3, 2),
	});

	EXPECT_DOUBLE_EQ(-9 * 4, r2.lower_);
	EXPECT_DOUBLE_EQ(6 * 4, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 3),
		estd::NumRange<double>(0.5, 2),
	});

	EXPECT_DOUBLE_EQ(0.5 * 4, r3.lower_);
	EXPECT_DOUBLE_EQ(6 * 4, r3.upper_);
}


TEST(STABILIZER, Conv)
{
	teq::Shape shape({2, 3, 4});
	teq::Shape kshape({3, 2});
	auto a = make_var(shape);
	auto k = make_var(kshape);

	auto setkey = eigen::Packer<teq::RanksT>::key_;
	marsh::NumArray<int64_t> numsobj({0, 2});

	auto f = make_fnc("", egen::CONV, teq::TensptrsT{a, k});
	EXPECT_CALL(*f, shape()).WillRepeatedly(Return(shape));

	EXPECT_CALL(*f, get_attr(setkey)).WillRepeatedly(Return(&numsobj));
	EXPECT_CALL(Const(*f), get_attr(setkey)).WillRepeatedly(Return(&numsobj));

	auto r1 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(-4, 2),
		estd::NumRange<double>(-3, 3),
	});

	EXPECT_DOUBLE_EQ(-12 * 6, r1.lower_);
	EXPECT_DOUBLE_EQ(12 * 6, r1.upper_);

	auto r2 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 3),
		estd::NumRange<double>(-3, 2),
	});

	EXPECT_DOUBLE_EQ(-9 * 6, r2.lower_);
	EXPECT_DOUBLE_EQ(6 * 6, r2.upper_);

	auto r3 = eigen::generate_range<double>(*f, {
		estd::NumRange<double>(1, 3),
		estd::NumRange<double>(0.5, 2),
	});

	EXPECT_DOUBLE_EQ(0.5 * 6, r3.lower_);
	EXPECT_DOUBLE_EQ(6 * 6, r3.upper_);
}


#endif // DISABLE_UTILS_STABILIZER_TEST
