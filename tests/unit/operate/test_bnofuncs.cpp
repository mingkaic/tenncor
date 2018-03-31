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


#ifndef DISABLE_BNOFUNCS_TEST // binary node functions


static const double ERR_THRESH = 0.08; // 8% error


struct BNOFUNCS : public testutils::fuzz_test {};


using namespace testutils;


using VARFUNC = std::function<nnet::varptr(std::vector<nnet::varptr>)>;


template <typename T>
using SVARFUNC = std::function<nnet::varptr(nnet::varptr,T)>;


template <typename T>
using SCALARS = std::function<T(T, T)>;


using TWODV = std::vector<std::vector<int64_t> >;


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


static inline TWODV create2D (std::vector<int64_t> juanD, size_t C, size_t R)
{
	TWODV res;
 	for (size_t y = 0; y < R; y++)
	{
		res.push_back(std::vector<int64_t>(C, 0));
	}

	for (size_t y = 0; y < R; y++)
	{
		for (size_t x = 0; x < C; x++)
		{
			size_t juan_coord = x + y * C;
			res[y][x] = juanD[juan_coord];
		}
	}
	return res;
}


static inline bool freivald (testutils::fuzz_test* fuzzer, TWODV a, TWODV b, TWODV c)
{
	assert(!b.empty());
	size_t rlen = b[0].size();
	// probability of false positive = 1/2^n
	// Pr(fp) = 0.1% ~~> n = 10
	size_t m = 10;
	for (size_t i = 0; i < m; i++)
	{
		// generate r of len b[0].size() or c[0].size()
		std::vector<size_t> r = fuzzer->get_int(rlen, nnutils::formatter() << "freivald_vec" << i, {0, 1});

		// p = a @ (b @ r) - c @ r
		std::vector<int64_t> br;
		for (size_t y = 0, n = b.size(); y < n; y++)
		{
			int64_t bri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				bri += b[y][x] * r[x];
			}
			br.push_back(bri);
		}

		std::vector<int64_t> cr;
		for (size_t y = 0, n = c.size(); y < n; y++)
		{
			int64_t cri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				cri += c[y][x] * r[x];
			}
			cr.push_back(cri);
		}

		std::vector<int64_t> p;
		size_t n = a.size();
		for (size_t y = 0; y < n; y++)
		{
			int64_t ari = 0;
			for (size_t x = 0, m = a[y].size(); x < m; x++)
			{
				ari += a[y][x] * br[x];
			}
			p.push_back(ari);
		}
		for (size_t j = 0; j < n; j++)
		{
			p[j] -= cr[j];
		}

		// if p != 0 -> return false
		if (!std::all_of(p.begin(), p.end(), [](int64_t d) { return d == 0; }))
		{
			return false;
		}
	}
	return true;
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
        EXPECT_EQ(nnet::constant::get<double>(1).get(), nnet::pow(nul, nul).get());
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
        EXPECT_EQ(nullptr, (nul + nul).get());
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
        EXPECT_EQ(nullptr, (nul - nul).get());
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
        EXPECT_EQ(nullptr, (nul * nul).get());
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
        EXPECT_THROW(nul / nul, std::logic_error);
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
        EXPECT_THROW(operator == (nul, nul), std::exception);
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
        EXPECT_THROW(operator != (nul, nul), std::exception);
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
        EXPECT_THROW(operator > (nul, nul), std::exception);
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
        EXPECT_THROW(operator < (nul, nul), std::exception);
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

	nnet::varptr res = nnet::binomial_sample(leaf0, leaf1);
	nnet::varptr resl = nnet::binomial_sample(scalar0, leaf1);
	nnet::varptr resr = nnet::binomial_sample(leaf0, scalar1);

	std::vector<uint64_t> output = nnet::expose<uint64_t>(res);
	std::vector<uint64_t> outputl = nnet::expose<uint64_t>(resl);
	std::vector<uint64_t> outputr = nnet::expose<uint64_t>(resr);

	// test behavior B000
	nnet::varptr res2 = nnet::binomial_sample(leaf0, leaf1);
	EXPECT_EQ(res.get(), res2.get());

	nnet::varptr opres = nnet::run_opcode({leaf0, leaf1}, BINO);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::binomial_sample(nul, leaf1).get());
	EXPECT_EQ(nullptr, nnet::binomial_sample(leaf0, nul).get());
	EXPECT_EQ(nullptr, nnet::binomial_sample(nul, nul).get());

	// approximate to normal distribution
	{
		std::vector<double> stdev_count(3, 0);
		for (size_t i = 0; i < n; ++i)
		{
			double mean = argument0[i] * argument1[i];
			double stdev = mean * (1 - argument1[i]);
			size_t index = std::abs(mean - output[i]) / stdev;
			if (index < stdev_count.size())
			{
				stdev_count[index]++;
			}
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
	//		if (index < stdev_count.size())
	//		{
	//			stdev_count[index]++;
	//		}
	// 	}
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
	// 	std::vector<double> stdev_count(3, 0);
	// 	for (size_t i = 0; i < n; ++i)
	// 	{
	// 		double mean = argument0[i] * scalar1;
	// 		double stdev = mean * (1 - scalar1);
	// 		size_t index = std::abs(mean - outputr[i]) / stdev;
	//		if (index < stdev_count.size())
	//		{
	//			stdev_count[index]++;
	//		}
	// 	}
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
	nnet::tensorshape shape = random_def_shape(this, {2, 12}, {10000, 78910});
	size_t n = shape.n_elems();
	std::vector<double> argument0 = get_double(n, "argument0", {-2, 1});
	std::vector<double> argument1 = get_double(n, "argument1", {2, 5});
	double scalar0 = get_double(1, "scalar0", {-2, 1})[0];
	double scalar1 = get_double(1, "scalar1", {2, 5})[0];
	nnet::varptr leaf0 = nnet::constant::get<double>(argument0, shape);
	nnet::varptr leaf1 = nnet::constant::get<double>(argument1, shape);

	nnet::varptr res = nnet::uniform_sample(leaf0, leaf1);
	nnet::varptr resl = nnet::uniform_sample(scalar0, leaf1);
	nnet::varptr resr = nnet::uniform_sample(leaf0, scalar1);

	std::vector<double> output = nnet::expose<double>(res);
	std::vector<double> outputl = nnet::expose<double>(resl);
	std::vector<double> outputr = nnet::expose<double>(resr);

	// test behavior B000
	nnet::varptr res2 = nnet::uniform_sample(leaf0, leaf1);
	EXPECT_EQ(res.get(), res2.get());

	nnet::varptr opres = nnet::run_opcode({leaf0, leaf1}, UNIF);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::uniform_sample(nul, leaf1).get());
	EXPECT_EQ(nullptr, nnet::uniform_sample(leaf0, nul).get());
	EXPECT_EQ(nullptr, nnet::uniform_sample(nul, nul).get());

	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_LE(argument0[i], output[i]);
		EXPECT_GE(argument1[i], output[i]);

		EXPECT_LE(scalar0, outputl[i]);
		EXPECT_GE(argument1[i], outputl[i]);
	
		EXPECT_LE(argument0[i], outputr[i]);
		EXPECT_GE(scalar1, outputr[i]);
	}
}


TEST_F(BNOFUNCS, Norm_B0xxAndB131)
{
	nnet::tensorshape shape = random_def_shape(this, {2, 12}, {10000, 78910});
	size_t n = shape.n_elems();
	std::vector<double> argument0 = get_double(n, "argument0", {-21, 154});
	std::vector<double> argument1 = get_double(n, "argument1", {1, 123});
	double scalar0 = get_double(1, "scalar0", {-21, 154})[0];
	double scalar1 = get_double(1, "scalar1", {1, 123})[0];
	nnet::varptr leaf0 = nnet::constant::get<double>(argument0, shape);
	nnet::varptr leaf1 = nnet::constant::get<double>(argument1, shape);

	nnet::varptr res = nnet::normal_sample(leaf0, leaf1);
	nnet::varptr resl = nnet::normal_sample(scalar0, leaf1);
	nnet::varptr resr = nnet::normal_sample(leaf0, scalar1);

	std::vector<double> output = nnet::expose<double>(res);
	std::vector<double> outputl = nnet::expose<double>(resl);
	std::vector<double> outputr = nnet::expose<double>(resr);

	// test behavior B000
	nnet::varptr res2 = nnet::normal_sample(leaf0, leaf1);
	EXPECT_EQ(res.get(), res2.get());

	nnet::varptr opres = nnet::run_opcode({leaf0, leaf1}, NORM);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::normal_sample(nul, leaf1).get());
	EXPECT_EQ(nullptr, nnet::normal_sample(leaf0, nul).get());
	EXPECT_EQ(nullptr, nnet::normal_sample(nul, nul).get());

	{
		std::vector<double> stdev_count(3, 0);
		for (size_t i = 0; i < n; ++i)
		{
			double mean = argument0[i];
			double stdev = argument1[i];
			size_t index = std::abs(mean - output[i]) / stdev;
			if (index < stdev_count.size())
			{
				stdev_count[index]++;
			}
		}
		// check the first 3 stdev
		double expect68 = stdev_count[0] / n; // expect ~68%
		double expect95 = (stdev_count[0] + stdev_count[1]) / n; // expect ~95%
		double expect99 = (stdev_count[0] + stdev_count[1] + stdev_count[2]) / n; // expect ~99.7%

		// allow larger error threshold to account for small n
		EXPECT_GT(ERR_THRESH, std::abs(0.68 - expect68));
		EXPECT_GT(ERR_THRESH, std::abs(0.95 - expect95));
		EXPECT_GT(ERR_THRESH, std::abs(0.997 - expect99));
	}

	{
		std::vector<double> stdev_count(3, 0);
		for (size_t i = 0; i < n; ++i)
		{
			double mean = scalar0;
			double stdev = argument1[i];
			size_t index = std::abs(mean - outputl[i]) / stdev;
			if (index < stdev_count.size())
			{
				stdev_count[index]++;
			}
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

	{
		std::vector<double> stdev_count(3, 0);
		for (size_t i = 0; i < n; ++i)
		{
			double mean = argument0[i];
			double stdev = scalar1;
			size_t index = std::abs(mean - outputr[i]) / stdev;
			if (index < stdev_count.size())
			{
				stdev_count[index]++;
			}
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
}


TEST_F(BNOFUNCS, Matmul_B0xxAndB132)
{
	std::vector<size_t> clist = random_def_shape(this); // <m, n, ...>
	size_t k = get_int(1, "k", {1, 8})[0];
	size_t m = clist[0];
	size_t n = clist[1];
	nnet::tensorshape shape0 = clist;
	clist[1] = clist[0];
	clist[0] = k;
	nnet::tensorshape shape1 = clist; // <k, m, ...>
	clist[1] = n;
	nnet::tensorshape outshape = clist; // <k, n, ...>
	size_t n1 = shape0.n_elems();
	size_t n2 = shape1.n_elems();
	size_t nout = outshape.n_elems();
	auto temp0 = get_int(n1, "argument0", {0, 19});
	auto temp1 = get_int(n2, "argument1", {0, 19});
	std::vector<int64_t> argument0(temp0.begin(), temp0.end());
	std::vector<int64_t> argument1(temp1.begin(), temp1.end());
	nnet::varptr leaf0 = nnet::constant::get<int64_t>(argument0, shape0);
	nnet::varptr leaf1 = nnet::constant::get<int64_t>(argument1, shape1);
	nnet::varptr res = nnet::matmul(leaf0, leaf1);

	nnet::tensor* ten = res->get_tensor();
	std::vector<int64_t> output = nnet::expose<int64_t>(res);
	ASSERT_EQ(nout, output.size());
	nnet::tensorshape shape = ten->get_shape();
	EXPECT_TRUE(tensorshape_equal(outshape, shape)) <<
		sprintf("expecting %p, got %p", &outshape, &shape);

	// test behavior B000
	nnet::varptr res2 = nnet::matmul(leaf0, leaf1);
	EXPECT_EQ(res.get(), res2.get());

	nnet::varptr opres = nnet::run_opcode({leaf0, leaf1}, MATMUL);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::matmul(nul, leaf1).get());
	EXPECT_EQ(nullptr, nnet::matmul(leaf0, nul).get());
	EXPECT_EQ(nullptr, nnet::matmul(nul, nul).get());

	size_t nchunks = std::accumulate(clist.begin() + 2, clist.end(), 1, std::multiplies<size_t>());
	size_t nchunk0 = m * n;
	size_t nchunk1 = k * m;
	size_t nchunkr = k * n;
	auto it0 = argument0.begin();
	auto it1 = argument1.begin();
	auto itr = output.begin();
	for (size_t i = 0; i < nchunks; ++i)
	{
		std::vector<int64_t> chunka(it0 + i * nchunk0, it0 + (i + 1) * nchunk0);
		std::vector<int64_t> chunkb(it1 + i * nchunk1, it1 + (i + 1) * nchunk1);
		std::vector<int64_t> chunkr(itr + i * nchunkr, itr + (i + 1) * nchunkr);
		EXPECT_TRUE(freivald(this, create2D(chunka, m, n), create2D(chunkb, k, m), create2D(chunkr, k, n))) <<
			testutils::sprintf("matrix multiplication failed at level %i", i);
	}
}


#endif /* DISABLE_BNOFUNCS_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
