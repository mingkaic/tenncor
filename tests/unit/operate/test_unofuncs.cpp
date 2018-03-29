//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATE_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"

#include "operate/operations.hpp"


#ifndef DISABLE_UNOFUNCS_TEST


class UNOFUNCS : public testify::fuzz_test {};


using namespace testutils;


using VARFUNC = std::function<nnet::varptr(std::vector<nnet::varptr>)>;


using SCALAR = std::function<double(double)>;


using AGGS = std::function<double(std::vector<double>)>;


template <typename T>
using SCALARS = std::function<T(T, T)>;


using ZCHECK = std::function<void(VARFUNC)>;


static void expect_zero (VARFUNC op)
{
	nnet::varptr goz = op({nnet::varptr()});
	EXPECT_EQ(nullptr, goz.get());
}


static void expect_wun (VARFUNC op)
{
	nnet::varptr goz = op({nnet::varptr()});
	ASSERT_NE(nullptr, goz.get());
	std::vector<double> data = nnet::expose<double>(goz);
	EXPECT_EQ(1, data.size());
	EXPECT_EQ(1, data[0]);
}


static void expect_throw (VARFUNC op)
{
	EXPECT_THROW(op({nnet::varptr()}), std::exception); // todo: specify logic error
}


static void unarNodeTest (testify::fuzz_test* fuzzer, OPCODE opcode, VARFUNC op, 
	SCALAR expect, ZCHECK exz, std::pair<double,double> limits = {-1, 1})
{
	nnet::tensorshape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument = fuzzer->get_double(n, "argument", limits);
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);

	// test behavior B000
	nnet::varptr res = op({leaf});
	nnet::varptr res2 = op({leaf});
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	exz(op);

	// test behavior B1xx
	nnet::tensor* ten = res->get_tensor();
	ASSERT_NE(nullptr, ten);
	EXPECT_TRUE(tensorshape_equal(shape, ten->get_shape()));
	std::vector<double> output = nnet::expose<double>(res.get());
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(expect(argument[i]), output[i]);
	}

	nnet::varptr opres = nnet::run_opcode({leaf}, opcode);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
}


TEST_F(UNOFUNCS, Abs_B0xxAndB100)
{
	unarNodeTest(this, ABS,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::abs(args[0]);
	}, 
	[](double arg) { return std::abs(arg); }, expect_zero);
}


TEST_F(UNOFUNCS, Neg_B0xxAndB101)
{
	unarNodeTest(this, NEG,
	[](std::vector<nnet::varptr> args)
	{
		return -args[0];
	}, 
	[](double arg) { return -arg; }, expect_zero);
}


TEST_F(UNOFUNCS, Not_B0xxAndB102)
{
	unarNodeTest(this, NOT,
	[](std::vector<nnet::varptr> args)
	{
		return !args[0];
	}, 
	[](double arg) { return !arg; }, expect_wun);
}


TEST_F(UNOFUNCS, Sin_B0xxAndB103)
{
	unarNodeTest(this, SIN,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::sin(args[0]);
	}, 
	[](double arg) { return std::sin(arg); }, expect_zero);
}


TEST_F(UNOFUNCS, Cos_B0xxAndB104)
{
	unarNodeTest(this, COS,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::cos(args[0]);
	}, 
	[](double arg) { return std::cos(arg); }, expect_wun);
}


TEST_F(UNOFUNCS, Tan_B0xxAndB105)
{
	unarNodeTest(this, TAN,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::tan(args[0]);
	}, 
	[](double arg) { return std::tan(arg); }, expect_zero);
}


TEST_F(UNOFUNCS, Exp_B0xxAndB106)
{
	unarNodeTest(this, EXP,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::exp(args[0]);
	}, 
	[](double arg) { return std::exp(arg); }, expect_wun);
}


TEST_F(UNOFUNCS, Log_B0xxAndB107)
{
	unarNodeTest(this, LOG,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::log(args[0]);
	},
	[](double arg) { return std::log(arg); }, expect_throw, {0.5, 7});
}


TEST_F(UNOFUNCS, Sqrt_B0xxAndB108)
{
	unarNodeTest(this, SQRT,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::sqrt(args[0]);
	},
	[](double arg) { return std::sqrt(arg); }, expect_zero, {0, 7});
}


TEST_F(UNOFUNCS, Round_B0xxAndB109)
{
	unarNodeTest(this, ROUND,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::round(args[0]);
	},
	[](double arg) { return std::round(arg); }, expect_zero);
}


#endif /* DISABLE_UNOFUNCS_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
