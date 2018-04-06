//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATE_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "sgen.hpp"
#include "print.hpp"

#include "operate/data_op.hpp"


#ifndef DISABLE_UBIND_TEST


class UBIND : public testutils::fuzz_test {};


using namespace testutils;


using SCALAR = std::function<double(double)>;


using AGGS = std::function<double(std::vector<double>)>;


static void unaryElemTest (testutils::fuzz_test* fuzzer, std::string op, 
	SCALAR expect, std::pair<double,double> limits = {-1, 1})
{
	nnet::tensorshape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument = fuzzer->get_double(n, "argument", limits);
	std::vector<double> output(n);

	ASSERT_TRUE(nnet::has_ele(op)) <<
		testutils::sprintf("unary %s not found", op.c_str());
	nnet::VTFUNC_F unfunc = nnet::ebind(op);

	unfunc(nnet::DOUBLE, nnet::VARR_T{(void*) &output[0], shape}, {nnet::CVAR_T{(const void*) &argument[0], shape}});

	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(expect(argument[i]), output[i]);
	}
}


static void unaryAggTest (testutils::fuzz_test* fuzzer, std::string op, 
	double init, AGGS agg, std::pair<double,double> limits = {-1, 1})
{
	nnet::tensorshape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument = fuzzer->get_double(n, "argument", limits);
	std::vector<double> output(n);

	ASSERT_TRUE(nnet::has_agg(op)) <<
		testutils::sprintf("aggregate %s not found", op.c_str());
	nnet::AFUNC_F afunc = nnet::abind(op)(nnet::DOUBLE);

	double expect = agg(argument);
	double result = init;
	const void* array = (const void*) &argument[0];
	for (size_t i = 0; i < argument.size(); i++)
	{
		afunc(i, (void*) &result, array);
	}
	EXPECT_EQ(expect, result);
}


TEST_F(UBIND, Abs_A000)
{
	unaryElemTest(this, "abs",
	[](double var) { return std::abs(var); });
}


TEST_F(UBIND, Neg_A001)
{
	unaryElemTest(this, "neg",
	[](double var) { return -var; });
}


TEST_F(UBIND, Not_A002)
{
	unaryElemTest(this, "logic_not",
	[](double var) { return !var; });
}


TEST_F(UBIND, Sin_A003)
{
	unaryElemTest(this, "sin",
	[](double var) { return std::sin(var); });
}


TEST_F(UBIND, Cos_A004)
{
	unaryElemTest(this, "cos",
	[](double var) { return std::cos(var); });
}


TEST_F(UBIND, Tan_A005)
{
	unaryElemTest(this, "tan",
	[](double var) { return std::tan(var); });
}


TEST_F(UBIND, Exp_A006)
{
	unaryElemTest(this, "exp",
	[](double var) { return std::exp(var); });
}


TEST_F(UBIND, Log_A007)
{
	unaryElemTest(this, "log",
	[](double var) { return std::log(var); }, {0.5, 7});
}


TEST_F(UBIND, Sqrt_A008)
{
	unaryElemTest(this, "sqrt",
	[](double var) { return std::sqrt(var); }, {0, 7});
}


TEST_F(UBIND, Round_A009)
{
	unaryElemTest(this, "round",
	[](double var) { return std::round(var); });
}


TEST_F(UBIND, Argmax_A010)
{
	unaryAggTest(this, "argmax", 0,
	[](std::vector<double> vec) -> double
	{
		auto it = std::max_element(vec.begin(), vec.end());
		return (double) std::distance(vec.begin(), it);
	});
}


TEST_F(UBIND, Max_A011)
{
	unaryAggTest(this, "max", std::numeric_limits<double>::min(),
	[](std::vector<double> vec) -> double
	{
		return *std::max_element(vec.begin(), vec.end());
	});
}


TEST_F(UBIND, Sum_A012)
{
	unaryAggTest(this, "sum", 0,
	[](std::vector<double> vec) -> double
	{
		return std::accumulate(vec.begin(), vec.end(), (double) 0);
	});
}


#endif /* DISABLE_UBIND_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
