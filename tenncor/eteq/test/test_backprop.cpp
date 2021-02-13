
#ifndef DISABLE_ETEQ_BACKPROP_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/eteq/eteq.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::Throw;


static void unary_derivative (egen::_GENERATED_OPCODE opcode, const std::string& graph)
{
	eteq::DerivativeFuncs der;

	teq::Shape shape({3, 2});
	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, shape, "super");
	auto arg = make_var(data.data(), devref, shape, "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = make_fnc("op", opcode, teq::TensptrsT{arg});
	EXPECT_CALL(*op, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*op, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(graph, result);
}


static void binary_derivative (egen::_GENERATED_OPCODE opcode, const std::string& graph)
{
	eteq::DerivativeFuncs der;

	teq::Shape shape({3, 2});
	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, shape, "super");
	auto arg = make_var(data.data(), devref, shape, "arg1");
	auto arg2 = make_var(data.data(), devref, shape, "arg2");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = make_fnc("op", opcode, teq::TensptrsT{arg,arg2});
	EXPECT_CALL(*op, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*op, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto result = der.lderive(op, super, 1);
	EXPECT_GRAPHEQ(graph, result);
}


TEST(BACKPROP, Passthrough)
{
	// Identity, Cast, Round, Add
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3, 2}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3, 2}), "arg1");
	auto arg2 = make_var(data.data(), devref, teq::Shape({3, 2}), "arg2");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = make_fnc("op", egen::IDENTITY, teq::TensptrsT{arg, arg2});
	EXPECT_CALL(*op, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto result = der.lderive(op, super, 1);
	EXPECT_EQ(super.get(), result.get());
}


TEST(BACKPROP, Neg)
{
	unary_derivative(egen::NEG,
		"(NEG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Tan)
{
	unary_derivative(egen::TAN,
		"(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(SQUARE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(COS<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Log)
{
	unary_derivative(egen::LOG,
		"(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Sqrt)
{
	unary_derivative(egen::SQRT,
		"(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(constant:2<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Abs)
{
	unary_derivative(egen::ABS,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Sin)
{
	unary_derivative(egen::SIN,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(COS<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Cos)
{
	unary_derivative(egen::COS,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(NEG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SIN<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Exp)
{
	unary_derivative(egen::EXP,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Square)
{
	unary_derivative(egen::SQUARE,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:2<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Cube)
{
	unary_derivative(egen::CUBE,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:3<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SQUARE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Sigmoid)
{
	unary_derivative(egen::SIGMOID,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SUB<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______|___`--(constant:1<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___________`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Tanh)
{
	unary_derivative(egen::TANH,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(SUB<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:1<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SQUARE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___________`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Pow)
{
	binary_derivative(egen::POW,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(LOG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Mul)
{
	binary_derivative(egen::MUL,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, MinMax)
{
	binary_derivative(egen::MAX,
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(EQ<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Sub)
{
	binary_derivative(egen::SUB,
		"(NEG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, Div)
{
	binary_derivative(egen::DIV,
		"(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(NEG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___|___`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n");
}


TEST(BACKPROP, ReduceSum)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::REDUCE_SUM,teq::TensptrsT{arg},std::set<teq::RankT>{1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\1\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, ReduceProd)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::REDUCE_PROD,teq::TensptrsT{arg},std::set<teq::RankT>{1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:super<DOUBLE>[3\\1\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(REDUCE_PROD<DOUBLE>[3\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, ReduceMinMax)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::REDUCE_MAX,teq::TensptrsT{arg},std::set<teq::RankT>{1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(EQ<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(REDUCE_MAX<DOUBLE>[3\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(constant:super<DOUBLE>[3\\1\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Extend)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3,2,4}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::EXTEND,teq::TensptrsT{arg},teq::DimsT{1, 1, 4});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(REDUCE_SUM<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\4\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Permute)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3,2}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::PERMUTE,teq::TensptrsT{arg},
		teq::RanksT{1, 2, 0});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(PERMUTE<DOUBLE>[1\\3\\2\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Reshape)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6, 7, 8};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({2,2,2}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({4,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::RESHAPE,teq::TensptrsT{arg},
		teq::Shape({2, 2, 2}));

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(RESHAPE<DOUBLE>[4\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[2\\2\\2\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Matmul)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({2,2}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	auto arg2 = make_var(data.data(), devref, teq::Shape({2,3}), "arg2");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::CONTRACT, teq::TensptrsT{arg, arg2},
		eigen::PairVecT<teq::RankT>{{0, 1}});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 1);
	EXPECT_GRAPHEQ(
		"(PERMUTE<DOUBLE>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_`--(CONTRACT<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:super<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Conv)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({2}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	auto arg2 = make_var(data.data(), devref, teq::Shape({2,2}), "arg2");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::CONV, teq::TensptrsT{arg, arg2},
		teq::RanksT{0, 1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 1);
	EXPECT_GRAPHEQ(
		"(CONV<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[2\\1\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Slice)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({2}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::SLICE, teq::TensptrsT{arg},
		eigen::PairVecT<teq::DimT>{{1, 2}, {0, 1}});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(PAD<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[2\\1\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Pad)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({4,3}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::PAD, teq::TensptrsT{arg},
		eigen::PairVecT<teq::DimT>{{1, 1}, {0, 1}});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(SLICE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[4\\3\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Concat)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3,4}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	auto arg2 = make_var(data.data(), devref, teq::Shape({3,2}), "arg2");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::CONCAT, teq::TensptrsT{arg, arg2}, teq::RankT(1));

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 1);
	EXPECT_GRAPHEQ(
		"(SLICE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\4\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Stride)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({1,2}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::STRIDE, teq::TensptrsT{arg}, teq::DimsT{2, 1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(SCATTER<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[1\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Scatter)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3,4}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::SCATTER, teq::TensptrsT{arg},
		teq::Shape({3, 4}), teq::DimsT{1, 2});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(STRIDE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\4\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Reverse)
{
	eteq::DerivativeFuncs der;

	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, teq::Shape({3,2}), "super");
	auto arg = make_var(data.data(), devref, teq::Shape({3,2}), "arg1");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = eteq::make_functor(egen::REVERSE, teq::TensptrsT{arg}, std::set<teq::RankT>{1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(REVERSE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Select)
{
	eteq::DerivativeFuncs der;

	teq::Shape shape({3,2});
	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, shape, "super");
	auto arg = make_var(data.data(), devref, shape, "arg1");
	auto arg2 = make_var(data.data(), devref, shape, "arg2");
	auto arg3 = make_var(data.data(), devref, shape, "arg3");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg3, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = make_fnc("op", egen::SELECT, teq::TensptrsT{arg, arg2, arg3});
	EXPECT_CALL(*op, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*op, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto result = der.lderive(op, super, 1);
	EXPECT_GRAPHEQ(
		"(SELECT<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:0<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Zeros)
{
	eteq::DerivativeFuncs der;

	teq::Shape shape({3,2});
	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, shape, "super");
	auto arg = make_var(data.data(), devref, shape, "arg1");
	auto arg2 = make_var(data.data(), devref, shape, "arg2");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = make_fnc("op", egen::RAND_UNIF, teq::TensptrsT{arg, arg2});
	EXPECT_CALL(*op, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*op, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	auto result = der.lderive(op, super, 1);
	EXPECT_GRAPHEQ(
		"(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:0<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Fatals)
{
	auto logger = new exam::MockLogger();
	global::set_logger(logger);
	EXPECT_CALL(*logger, supports_level(logs::fatal_level)).WillRepeatedly(Return(true));

	eteq::DerivativeFuncs der;

	teq::Shape shape({3, 2});
	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto super = make_var(data.data(), devref, shape, "super");
	auto arg = make_var(data.data(), devref, shape, "arg1");
	auto arg2 = make_var(data.data(), devref, shape, "arg2");
	EXPECT_CALL(*super, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));
	auto op = make_fnc("op", egen::ASSIGN, teq::TensptrsT{arg, arg2});
	EXPECT_CALL(*op, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*op, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	std::string fatalmsg = "Unsupported op derivation op";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(der.lderive(op, super, 1), fatalmsg.c_str());

	auto op2 = make_fnc("zop", 999999, teq::TensptrsT{arg, arg2});
	EXPECT_CALL(*op2, shape()).WillRepeatedly(Return(shape));
	EXPECT_CALL(*op2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));

	std::string fatalmsg1 = "Unsupported op derivation zop";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(der.lderive(op2, super, 1), fatalmsg1.c_str());

	global::set_logger(new exam::NoSupportLogger());
}


TEST(BACKPROP, ZeroOnes)
{
	MockMeta mockmeta;
	MockLeaf t1;
	MockLeaf t2;
	make_var(t1, teq::Shape({1, 2, 3}), "t1");
	make_var(t2, teq::Shape({3, 2, 4}), "t2");
	EXPECT_CALL(t1, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(t2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("FLOAT"));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::FLOAT));
	eteq::DerivativeFuncs der;

	auto result = der.get_const_zero(t1);
	ASSERT_NE(nullptr, result);
	EXPECT_GRAPHEQ("(constant:0<FLOAT>[1\\2\\3\\1\\1\\1\\1\\1])\n", result);

	auto result2 = der.get_const_one(t2);
	ASSERT_NE(nullptr, result2);
	EXPECT_GRAPHEQ("(constant:1<FLOAT>[3\\2\\4\\1\\1\\1\\1\\1])\n", result2);
}


TEST(BACKPROP, AddHelper)
{
	eteq::DerivativeFuncs der;

	teq::Shape shape({3, 2});
	std::vector<double> data{1, 2, 3, 4, 5, 6};
	MockDeviceRef devref;
	MockMeta mockmeta;
	auto arg = make_var(data.data(), devref, shape, "arg1");
	auto arg2 = make_var(data.data(), devref, shape, "arg2");
	auto arg3 = make_var(data.data(), devref, shape, "arg3");
	EXPECT_CALL(*arg, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg2, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(*arg3, get_meta()).WillRepeatedly(ReturnRef(mockmeta));
	EXPECT_CALL(mockmeta, type_code()).WillRepeatedly(Return(egen::DOUBLE));
	EXPECT_CALL(mockmeta, type_label()).WillRepeatedly(Return("DOUBLE"));

	auto result = der.add({arg, arg2, arg3});
	EXPECT_GRAPHEQ(
		"(ADD<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg3<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


#endif // DISABLE_ETEQ_BACKPROP_TEST
