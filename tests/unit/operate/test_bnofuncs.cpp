//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATE_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "fuzz.hpp"
#include "sgen.hpp"
#include "check.hpp"
#include "print.hpp"

#include "operate/operations.hpp"


#ifndef DISABLE_BNOFUNCS_TEST


#define ERR_THRESH 0.08 // 8% error


struct BNOFUNCS : public testutils::fuzz_test {};


using namespace testutils;


using VARFUNC = std::function<nnet::varptr(std::vector<nnet::varptr>)>;


template <typename T>
using SVARFUNC = std::function<nnet::varptr(nnet::varptr,T)>;


template <typename T>
using SCALARS = std::function<T(T, T)>;


using ZCHECK = std::function<void(std::vector<nnet::varptr>)>;


static void binarNodeTest (testutils::fuzz_test* fuzzer, OPCODE opcode, VARFUNC op,
    SVARFUNC<double> op_s1, SVARFUNC<double> op_s2, SCALARS<double> expect, ZCHECK exz, 
    std::pair<double,double> limits = {-1, 1})
{
	nnet::tensorshape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument0 = fuzzer->get_double(n, "argument0", limits);
	std::vector<double> argument1 = fuzzer->get_double(n, "argument1", limits);
	std::vector<double> scalars = fuzzer->get_double(2, "scalars", limits);
	nnet::varptr leaf0 = nnet::constant::get<double>(argument0, shape);
	nnet::varptr leaf1 = nnet::constant::get<double>(argument1, shape);

	nnet::varptr res = op({leaf0, leaf1});
	nnet::varptr resl = op_s1(leaf1, scalars[0]);
	nnet::varptr resr = op_s2(leaf0, scalars[1]);

	// test behavior B000
	nnet::varptr res2 = op({leaf0, leaf1});
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	exz({leaf0, leaf1});

	// test behavior B1xx
	nnet::tensor* ten = res->get_tensor();
	ASSERT_NE(nullptr, ten);
	EXPECT_TRUE(tensorshape_equal(shape, ten->get_shape()));
	std::vector<double> output = nnet::expose<double>(res.get());
	std::vector<double> outputl = nnet::expose<double>(resl.get());
	std::vector<double> outputr = nnet::expose<double>(resr.get());
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(expect(argument0[i], argument1[i]), output[i]);
		EXPECT_EQ(expect(scalars[0], argument1[i]), outputl[i]);
		EXPECT_EQ(expect(argument0[i], scalars[1]), outputr[i]);
	}

	nnet::varptr opres = nnet::run_opcode({leaf0, leaf1}, opcode);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
}


static void binarIntNodeTest (testutils::fuzz_test* fuzzer, OPCODE opcode, VARFUNC op,
	SVARFUNC<uint64_t> op_s1, SVARFUNC<uint64_t> op_s2, SCALARS<uint64_t> expect, ZCHECK exz, 
    std::pair<uint64_t,uint64_t> limits = {0, 12})
{
	nnet::tensorshape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	auto temp0 = fuzzer->get_int(n, "argument0", limits);
	auto temp1 = fuzzer->get_int(n, "argument1", limits);
	auto temp2 = fuzzer->get_int(2, "scalars", limits);
	std::vector<uint64_t> argument0(temp0.begin(), temp0.end());
	std::vector<uint64_t> argument1(temp1.begin(), temp1.end());
	std::vector<uint64_t> scalars(temp2.begin(), temp2.end());
	nnet::varptr leaf0 = nnet::constant::get<uint64_t>(argument0, shape);
	nnet::varptr leaf1 = nnet::constant::get<uint64_t>(argument1, shape);

	nnet::varptr res = op({leaf0, leaf1});
	nnet::varptr resl = op_s1(leaf1, scalars[0]);
	nnet::varptr resr = op_s2(leaf0, scalars[1]);

	// test behavior B000
	nnet::varptr res2 = op({leaf0, leaf1});
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	exz({leaf0, leaf1});

	// test behavior B1xx
	nnet::tensor* ten = res->get_tensor();
	ASSERT_NE(nullptr, ten);
	EXPECT_TRUE(tensorshape_equal(shape, ten->get_shape()));
	std::vector<uint64_t> output = nnet::expose<uint64_t>(res.get());
	std::vector<uint64_t> outputl = nnet::expose<uint64_t>(resl.get());
	std::vector<uint64_t> outputr = nnet::expose<uint64_t>(resr.get());
	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_EQ(expect(argument0[i], argument1[i]), output[i]);
		EXPECT_EQ(expect(scalars[0], argument1[i]), outputl[i]);
		EXPECT_EQ(expect(argument0[i], scalars[1]), outputr[i]);
	}

	nnet::varptr opres = nnet::run_opcode({leaf0, leaf1}, opcode);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
}


TEST_F(BNOFUNCS, Pow_B0xxAndB120)
{
	binarNodeTest(this, POW,
	[](std::vector<nnet::varptr> args)
	{
		return nnet::pow(args[0], args[1]);
	},
	[](nnet::varptr b, double a)
	{
		return nnet::pow(a, b);
	},
	[](nnet::varptr a, double b)
	{
		return nnet::pow(a, b);
	},
	[](double a, double b) { return std::pow(a, b); },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
        EXPECT_EQ(nnet::constant::get<double>(1).get(), nnet::pow(args[0], nul).get());
        EXPECT_EQ(nullptr, nnet::pow(nul, args[1]).get());
    }, {0, 7});
}


TEST_F(BNOFUNCS, Add_B0xxAndB121)
{
	binarNodeTest(this, ADD,
	[](std::vector<nnet::varptr> args)
	{
		return args[0] + args[1];
	},
	[](nnet::varptr b, double a)
	{
        return a + b;
	},
	[](nnet::varptr a, double b)
	{
        return a + b;
	},
	[](double a, double b) { return a + b; },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
        EXPECT_EQ(args[0].get(), (args[0] + nul).get());
        EXPECT_EQ(args[1].get(), (nul + args[1]).get());
    });
}


TEST_F(BNOFUNCS, Sub_B0xxAndB122)
{
	binarNodeTest(this, SUB,
	[](std::vector<nnet::varptr> args)
	{
		return args[0] - args[1];
	},
	[](nnet::varptr b, double a)
	{
        return a - b;
	},
	[](nnet::varptr a, double b)
	{
        return a - b;
	},
	[](double a, double b) { return a - b; },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
        EXPECT_EQ(args[0].get(), (args[0] - nul).get());
        EXPECT_EQ((-args[1]).get(), (nul - args[1]).get());
    });
}


TEST_F(BNOFUNCS, Mul_B0xxAndB123)
{
	binarNodeTest(this, MUL,
	[](std::vector<nnet::varptr> args)
	{
		return args[0] * args[1];
	},
	[](nnet::varptr b, double a)
	{
        return a * b;
	},
	[](nnet::varptr a, double b)
	{
        return a * b;
	},
	[](double a, double b) { return a * b; },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
        EXPECT_EQ(nullptr, (args[0] * nul).get());
        EXPECT_EQ(nullptr, (nul * args[1]).get());
    });
}


TEST_F(BNOFUNCS, Div_B0xxAndB124)
{
	binarNodeTest(this, DIV,
	[](std::vector<nnet::varptr> args)
	{
		return args[0] / args[1];
	},
	[](nnet::varptr b, double a)
	{
        return a / b;
	},
	[](nnet::varptr a, double b)
	{
        return a / b;
	},
	[](double a, double b) { return a / b; },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
        EXPECT_THROW(args[0] / nul, std::logic_error);
        EXPECT_EQ(nullptr, (nul / args[1]).get());
    });
}


TEST_F(BNOFUNCS, Eq_B0xxAndB125)
{
	binarIntNodeTest(this, EQ,
	[](std::vector<nnet::varptr> args)
	{
		return args[0] == args[1];
	},
	[](nnet::varptr b, uint64_t a)
	{
        return a == b;
	},
	[](nnet::varptr a, uint64_t b)
	{
        return a == b;
	},
	[](uint64_t a, uint64_t b) { return a == b; },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
		EXPECT_THROW(operator == (args[0], nul), std::exception);
		EXPECT_THROW(operator == (nul, args[1]), std::exception);
    });
}


TEST_F(BNOFUNCS, Neq_B0xxAndB126)
{
	binarIntNodeTest(this, NE,
	[](std::vector<nnet::varptr> args)
	{
		return args[0] != args[1];
	},
	[](nnet::varptr b, uint64_t a)
	{
        return a != b;
	},
	[](nnet::varptr a, uint64_t b)
	{
        return a != b;
	},
	[](uint64_t a, uint64_t b) { return a != b; },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
		EXPECT_THROW(operator != (args[0], nul), std::exception);
		EXPECT_THROW(operator != (nul, args[1]), std::exception);
    });
}


TEST_F(BNOFUNCS, Gt_B0xxAndB127)
{
	binarIntNodeTest(this, GT,
	[](std::vector<nnet::varptr> args)
	{
		return args[0] > args[1];
	},
	[](nnet::varptr b, uint64_t a)
	{
        return a > b;
	},
	[](nnet::varptr a, uint64_t b)
	{
        return a > b;
	},
	[](uint64_t a, uint64_t b) { return a > b; },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
		EXPECT_THROW(operator > (args[0], nul), std::exception);
		EXPECT_THROW(operator > (nul, args[1]), std::exception);
    });
}


TEST_F(BNOFUNCS, Lt_B0xxAndB128)
{
	binarIntNodeTest(this, LT,
	[](std::vector<nnet::varptr> args)
	{
		return args[0] < args[1];
	},
	[](nnet::varptr b, uint64_t a)
	{
		return a < b;
	},
	[](nnet::varptr a, uint64_t b)
	{
		return a < b;
	},
	[](uint64_t a, uint64_t b) { return a < b; },
    [](std::vector<nnet::varptr> args)
    {
		nnet::varptr nul;
		EXPECT_THROW(operator < (args[0], nul), std::exception);
		EXPECT_THROW(operator < (nul, args[1]), std::exception);
    });
}


TEST_F(BNOFUNCS, Binom_B0xxAndB129)
{
	nnet::tensorshape shape = random_def_shape(this, {2, 12}, {10000, 78910});
	size_t n = shape.n_elems();
	auto temp0 = get_int(n, "argument0", {2, 19});
	std::vector<uint64_t> argument0(temp0.begin(), temp0.end());
	std::vector<double> argument1 = get_double(n, "argument1", {0, 1});
	uint64_t scalar0 = get_int(1, "scalar0", {0, 12})[0];
	double scalar1 = get_double(1, "scalar1", {0, 1})[0];
	nnet::varptr leaf0 = nnet::constant::get<uint64_t>(argument0, shape);
	nnet::varptr leaf1 = nnet::constant::get<double>(argument1, shape);

	nnet::varptr res = binomial_sample(leaf0, leaf1);
	nnet::varptr resl = binomial_sample(scalar0, leaf1);
	nnet::varptr resr = binomial_sample(leaf0, scalar1);

	// test behavior B000
	nnet::varptr res2 = binomial_sample(leaf0, leaf1);
	EXPECT_EQ(res.get(), res2.get());

	nnet::varptr opres = nnet::run_opcode({leaf0, leaf1}, BINO);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, binomial_sample(nul, leaf1).get());
	EXPECT_EQ(nullptr, binomial_sample(leaf0, nul).get());

	std::vector<uint64_t> output = nnet::expose<uint64_t>(res);
	std::vector<uint64_t> outputl = nnet::expose<uint64_t>(resl);
	std::vector<uint64_t> outputr = nnet::expose<uint64_t>(resr);

	// approximate to normal distribution
	{
		std::vector<double> stdev_count(3, 0);
		for (size_t i = 0; i < n; ++i)
		{
			double mean = argument0[i] * argument1[i];
			double stdev = mean * (1 - argument1[i]);
			size_t index = std::abs(mean - output[i]) / stdev;
			if (index >= stdev_count.size())
			{
				stdev_count.insert(stdev_count.end(),
					index - stdev_count.size() + 1, 0);
			}
			assert(index < stdev_count.size());
			stdev_count[index]++;
		}
		// check the first 3 stdev
		double expect68 = stdev_count[0] / n; // expect ~68%
		double expect95 = (stdev_count[0] + stdev_count[1]) / n; // expect ~95%
		double expect99 = (stdev_count[0] + stdev_count[1] + stdev_count[2]) / n; // expect ~99.7%

		// allow larger error threshold to account for small n
		EXPECT_GT(ERR_THRESH * 2, std::abs(0.68 - expect68));
		EXPECT_GT(ERR_THRESH * 2, std::abs(0.95 - expect95));
		EXPECT_GT(ERR_THRESH * 2, std::abs(0.997 - expect99));
	}

	// todo: implement better test (see Hypothesis Test)
	// {
	// 	std::vector<double> stdev_count(3, 0);
	// 	for (size_t i = 0; i < n; ++i)
	// 	{
	// 		double mean = scalar0 * argument1[i];
	// 		double stdev = mean * (1 - argument1[i]);
	// 		size_t index = std::abs(mean - outputl[i]) / stdev;
	// 		if (index >= stdev_count.size())
	// 		{
	// 			stdev_count.insert(stdev_count.end(),
	// 				index - stdev_count.size() + 1, 0);
	// 		}
	// 		assert(index < stdev_count.size());
	// 		stdev_count[index]++;
	// 	}
	// 	std::cout << stdev_count.size() << std::endl;
	// 	// check the first 3 stdev
	// 	double expect68 = stdev_count[0] / n; // expect ~68%
	// 	double expect95 = (stdev_count[0] + stdev_count[1]) / n; // expect ~95%
	// 	double expect99 = (stdev_count[0] + stdev_count[1] + stdev_count[2]) / n; // expect ~99.7%

	// 	// allow larger error threshold to account for small n
	// 	EXPECT_GT(ERR_THRESH * 2, std::abs(0.68 - expect68));
	// 	EXPECT_GT(ERR_THRESH * 2, std::abs(0.95 - expect95));
	// 	EXPECT_GT(ERR_THRESH * 2, std::abs(0.997 - expect99));
	// }

	// {
	// 	std::cout << scalar1 << std::endl;
	// 	std::vector<double> stdev_count(3, 0);
	// 	for (size_t i = 0; i < n; ++i)
	// 	{
	// 		double mean = argument0[i] * scalar1;
	// 		double stdev = mean * (1 - scalar1);
	// 		size_t index = std::abs(mean - outputr[i]) / stdev;
	// 		if (index >= stdev_count.size())
	// 		{
	// 			stdev_count.insert(stdev_count.end(),
	// 				index - stdev_count.size() + 1, 0);
	// 		}
	// 		assert(index < stdev_count.size());
	// 		stdev_count[index]++;
	// 	}
	// 	std::cout << stdev_count.size() << std::endl;
	// 	// check the first 3 stdev
	// 	double expect68 = stdev_count[0] / n; // expect ~68%
	// 	double expect95 = (stdev_count[0] + stdev_count[1]) / n; // expect ~95%
	// 	double expect99 = (stdev_count[0] + stdev_count[1] + stdev_count[2]) / n; // expect ~99.7%

	// 	// allow larger error threshold to account for small n
	// 	EXPECT_GT(ERR_THRESH * 2, std::abs(0.68 - expect68));
	// 	EXPECT_GT(ERR_THRESH * 2, std::abs(0.95 - expect95));
	// 	EXPECT_GT(ERR_THRESH * 2, std::abs(0.997 - expect99));
	// }
}


TEST_F(BNOFUNCS, Unif_B0xxAndB130)
{
}


TEST_F(BNOFUNCS, Norm_B0xxAndB131)
{
}


#endif /* DISABLE_BNOFUNCS_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
