
#ifndef DISABLE_ETEQ_BACKPROP_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/eteq/eteq.hpp"


using ::testing::_;
using ::testing::Return;
using ::testing::Throw;


TEST(BACKPROP, Passthrough)
{
	// Identity, Cast, Round, Add
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"op", egen::IDENTITY});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 1);
	EXPECT_EQ(super.get(), result.get());
}


TEST(BACKPROP, Neg)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::NEG});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(NEG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Tan)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::TAN});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(SQUARE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(COS<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Log)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::LOG});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Sqrt)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::SQRT});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(constant:2<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Abs)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::ABS});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Sin)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::SIN});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(COS<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Cos)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::COS});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(NEG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SIN<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Exp)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::EXP});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Square)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::SQUARE});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:2<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Cube)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::CUBE});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:3<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SQUARE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Sigmoid)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::SIGMOID});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SUB<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______|___`--(constant:1<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___________`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Tanh)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg}, teq::Opcode{"op", egen::TANH});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 0);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(SUB<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:1<DOUBLE>[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(SQUARE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___________`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Pow)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"op", egen::POW});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 1);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(LOG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Mul)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"op", egen::MUL});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 1);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, MinMax)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"op", egen::MAX});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 1);
	EXPECT_GRAPHEQ(
		"(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(EQ<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(op<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Sub)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"op", egen::SUB});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 1);
	EXPECT_GRAPHEQ(
		"(NEG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Div)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"op", egen::DIV});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	auto result = der.lderive(op, super, 1);
	EXPECT_GRAPHEQ(
		"(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(MUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(NEG<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___|___`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, ReduceSum)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = eteq::make_functor(egen::REDUCE_SUM,teq::TensptrsT{arg},
		std::set<teq::RankT>{1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(EXTEND<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\1\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, ReduceProd)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = eteq::make_functor(egen::REDUCE_PROD,teq::TensptrsT{arg},
		std::set<teq::RankT>{1});

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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = eteq::make_functor(egen::REDUCE_MAX,teq::TensptrsT{arg},
		std::set<teq::RankT>{1});

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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2, 4}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = eteq::make_functor(egen::EXTEND,teq::TensptrsT{arg},
		teq::DimsT{1, 1, 4});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(REDUCE_SUM<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\4\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Permute)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8},
		teq::Shape({2, 2, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8},
		teq::Shape({4, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4},
		teq::Shape({2, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({2, 3}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = eteq::make_functor(egen::MATMUL, teq::TensptrsT{arg, arg2},
		eigen::PairVecT<teq::RankT>{{0, 1}});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 1);
	EXPECT_GRAPHEQ(
		"(PERMUTE<DOUBLE>[2\\3\\1\\1\\1\\1\\1\\1])\n"
		"_`--(MATMUL<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:super<DOUBLE>[2\\2\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Conv)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2},
		teq::Shape({2, 1}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4},
		teq::Shape({2, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2},
		teq::Shape({2, 1}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
		teq::Shape({4, 3}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6},
		teq::Shape({3, 4}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = eteq::make_functor(egen::CONCAT, teq::TensptrsT{arg, arg2}, teq::RankT(1));

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 1);
	EXPECT_GRAPHEQ(
		"(SLICE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\4\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Stride)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2},
		teq::Shape({1, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = eteq::make_functor(egen::STRIDE, teq::TensptrsT{arg}, teq::DimsT{2, 1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(SCATTER<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[1\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Scatter)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 4}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto op = eteq::make_functor(egen::REVERSE, teq::TensptrsT{arg}, std::set<teq::RankT>{1});

	auto result = der.lderive(std::dynamic_pointer_cast<teq::iFunctor>(op), super, 0);
	EXPECT_GRAPHEQ(
		"(REVERSE<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:super<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


TEST(BACKPROP, Select)
{
	eteq::DerivativeFuncs der;

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto arg3 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg3");
	arg3->meta_.tcode_ = egen::DOUBLE;
	arg3->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2, arg3}, teq::Opcode{"op", egen::SELECT});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"op", egen::RAND_UNIF});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

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

	auto super = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "super");
	super->meta_.tcode_ = egen::DOUBLE;
	super->meta_.tname_ = "DOUBLE";
	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto op = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"op", egen::ASSIGN});
	op->meta_.tcode_ = egen::DOUBLE;
	op->meta_.tname_ = "DOUBLE";

	std::string fatalmsg = "cannot derive op";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg)));
	EXPECT_FATAL(der.lderive(op, super, 1), fatalmsg.c_str());

	auto op2 = std::make_shared<MockFunctor>(
		teq::TensptrsT{arg, arg2}, teq::Opcode{"zop", 999999});
	op2->meta_.tcode_ = egen::DOUBLE;
	op2->meta_.tname_ = "DOUBLE";

	std::string fatalmsg1 = "Unknown op zop";
	EXPECT_CALL(*logger, log(logs::fatal_level, fatalmsg1, _)).Times(1).WillOnce(Throw(exam::TestException(fatalmsg1)));
	EXPECT_FATAL(der.lderive(op2, super, 1), fatalmsg1.c_str());

	global::set_logger(new exam::NoSupportLogger());
}


TEST(BACKPROP, ZeroOnes)
{
	eteq::DerivativeFuncs der;

	auto result = der.get_const_zero(teq::Shape({1, 2, 3}));
	EXPECT_GRAPHEQ("(constant:0<FLOAT>[1\\2\\3\\1\\1\\1\\1\\1])\n", result);

	auto result2 = der.get_const_one(teq::Shape({3, 2, 4}));
	EXPECT_GRAPHEQ("(constant:1<FLOAT>[3\\2\\4\\1\\1\\1\\1\\1])\n", result2);
}


TEST(BACKPROP, AddHelper)
{
	eteq::DerivativeFuncs der;

	auto arg = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg1");
	arg->meta_.tcode_ = egen::DOUBLE;
	arg->meta_.tname_ = "DOUBLE";
	auto arg2 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg2");
	arg2->meta_.tcode_ = egen::DOUBLE;
	arg2->meta_.tname_ = "DOUBLE";
	auto arg3 = std::make_shared<MockLeaf>(
		std::vector<double>{1, 2, 3, 4, 5, 6},
		teq::Shape({3, 2}), "arg3");
	arg3->meta_.tcode_ = egen::DOUBLE;
	arg3->meta_.tname_ = "DOUBLE";

	auto result = der.add({arg, arg2, arg3});
	EXPECT_GRAPHEQ(
		"(ADD<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg1<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg2<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n"
		"_`--(constant:arg3<DOUBLE>[3\\2\\1\\1\\1\\1\\1\\1])\n", result);
}


#endif // DISABLE_ETEQ_BACKPROP_TEST
