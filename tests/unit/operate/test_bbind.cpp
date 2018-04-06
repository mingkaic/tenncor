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


#ifndef DISABLE_BBIND_TEST


static const double ERR_THRESH = 0.08; // 8% error


class BBIND : public testutils::fuzz_test {};


using namespace testutils;


template <typename T>
using SCALARS = std::function<T(T, T)>;


using TWODV = std::vector<std::vector<int64_t> >;


static void binaryElemTest (testutils::fuzz_test* fuzzer, std::string op, 
	SCALARS<double> expect, std::pair<double,double> limits = {-1, 1})
{
	nnet::tensorshape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	std::vector<double> argument0 = fuzzer->get_double(n, "argument0", limits);
	std::vector<double> argument1 = fuzzer->get_double(n, "argument1", limits);
	std::vector<double> output(n);

	ASSERT_TRUE(nnet::has_ele(op)) <<
		testutils::sprintf("binary %s not found", op.c_str());
	nnet::VTFUNC_F bifunc = nnet::ebind(op);

	bifunc(nnet::DOUBLE, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	});

	for (size_t i = 0; i < n; ++i)
	{
		double expectd = expect(argument0[i], argument1[i]);
		EXPECT_EQ(expectd, output[i]);
	}
}


static void binaryElemTestInt (testutils::fuzz_test* fuzzer, std::string op, 
	SCALARS<uint64_t> expect, std::pair<uint64_t,uint64_t> limits = {0, 2})
{
	nnet::tensorshape shape = random_def_shape(fuzzer);
	size_t n = shape.n_elems();
	auto temp0 = fuzzer->get_int(n, "argument0", limits);
	auto temp1 = fuzzer->get_int(n, "argument1", limits);
	std::vector<uint64_t> argument0(temp0.begin(), temp0.end());
	std::vector<uint64_t> argument1(temp1.begin(), temp1.end());
	std::vector<uint64_t> output(n);

	ASSERT_TRUE(nnet::has_ele(op)) <<
		testutils::sprintf("binary %s not found", op.c_str());
	nnet::VTFUNC_F bifunc = nnet::ebind(op);

	bifunc(nnet::UINT64, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	});

	for (size_t i = 0; i < n; ++i)
	{
		uint64_t expectd = expect(argument0[i], argument1[i]);
		EXPECT_EQ(expectd, output[i]);
	}
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


TEST_F(BBIND, Pow_A020)
{
	binaryElemTest(this, "pow",
	[](double a, double b) { return std::pow(a, b); }, {0, 7});
}


TEST_F(BBIND, Add_A021)
{
	binaryElemTest(this, "add",
	[](double a, double b) { return a + b; });
}


TEST_F(BBIND, Sub_A022)
{
	binaryElemTest(this, "sub",
	[](double a, double b) { return a - b; });
}


TEST_F(BBIND, Mul_A023)
{
	binaryElemTest(this, "mul",
	[](double a, double b) { return a * b; });
}


TEST_F(BBIND, Div_A024)
{
	binaryElemTest(this, "div",
	[](double a, double b) { return a / b; });
}


TEST_F(BBIND, Eq_A025)
{
	binaryElemTestInt(this, "eq",
	[](uint64_t a, uint64_t b) -> uint64_t { return a == b; });
}


TEST_F(BBIND, Neq_A026)
{
	binaryElemTestInt(this, "neq",
	[](uint64_t a, uint64_t b) -> uint64_t { return a != b; });
}


TEST_F(BBIND, Lt_A027)
{
	binaryElemTestInt(this, "lt",
	[](uint64_t a, uint64_t b) -> uint64_t { return a < b; });
}


TEST_F(BBIND, Gt_A028)
{
	binaryElemTestInt(this, "gt",
	[](uint64_t a, uint64_t b) -> uint64_t { return a > b; });
}


TEST_F(BBIND, Uniform_A029)
{
	nnet::tensorshape shape = random_def_shape(this);
	size_t n = shape.n_elems();
	std::vector<double> argument0 = get_double(n, "argument0", {-2, 1});
	std::vector<double> argument1 = get_double(n, "argument1", {2, 5});
	std::vector<double> output(n);

	std::string opname = "rand_uniform";
	ASSERT_TRUE(nnet::has_ele(opname));
	nnet::VTFUNC_F bifunc = nnet::ebind(opname);

	bifunc(nnet::DOUBLE, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	});

	for (size_t i = 0; i < n; ++i)
	{
		EXPECT_LT(argument0[i], output[i]);
		EXPECT_GT(argument1[i], output[i]);
	}
}


TEST_F(BBIND, Binom_A030)
{
	nnet::tensorshape shape = random_def_shape(this, {2, 12}, {10000, 78910});
	size_t n = shape.n_elems();
	auto temp0 = get_int(n, "argument0", {2, 19});
	std::vector<uint64_t> argument0(temp0.begin(), temp0.end());
	std::vector<double> argument1 = get_double(n, "argument1", {0, 1});
	std::vector<uint64_t> output(n);

	std::string opname = "rand_binom";
	ASSERT_TRUE(nnet::has_ele(opname));
	nnet::VTFUNC_F bifunc = nnet::ebind(opname);

	EXPECT_THROW(bifunc(nnet::DOUBLE, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	EXPECT_THROW(bifunc(nnet::FLOAT, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	bifunc(nnet::UINT64, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	});

	// approximate to normal distribution
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

	double err1 = std::abs(0.68 - expect68);
	double err2 = std::abs(0.95 - expect95);
	double err3 = std::abs(0.997 - expect99);

	// allow larger error threshold to account for small n
	EXPECT_GT(ERR_THRESH * 2, err1);
	EXPECT_GT(ERR_THRESH * 2, err2);
	EXPECT_GT(ERR_THRESH * 2, err3);
}


TEST_F(BBIND, Norm_A031)
{
	nnet::tensorshape shape = random_def_shape(this, {2, 12}, {10000, 78910});
	size_t n = shape.n_elems();
	std::vector<double> argument0 = get_double(n, "argument0", {-21, 154});
	std::vector<double> argument1 = get_double(n, "argument1", {1, 123});
	std::vector<double> output(n);

	std::string opname = "rand_normal";
	ASSERT_TRUE(nnet::has_ele(opname));
	nnet::VTFUNC_F bifunc = nnet::ebind(opname);

	EXPECT_THROW(bifunc(nnet::INT8, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	EXPECT_THROW(bifunc(nnet::UINT8, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	EXPECT_THROW(bifunc(nnet::INT16, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	EXPECT_THROW(bifunc(nnet::UINT16, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	EXPECT_THROW(bifunc(nnet::INT32, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	EXPECT_THROW(bifunc(nnet::UINT32, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	EXPECT_THROW(bifunc(nnet::INT64, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	EXPECT_THROW(bifunc(nnet::UINT64, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	}), std::bad_function_call);

	bifunc(nnet::DOUBLE, nnet::VARR_T{(void*) &output[0], shape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape},
		nnet::CVAR_T{(const void*) &argument1[0], shape},
	});

	std::vector<double> stdev_count(3, 0);
	for (size_t i = 0; i < n; ++i)
	{
		size_t index = std::abs(argument0[i] - output[i]) / argument1[i];
		if (index < stdev_count.size())
		{
			stdev_count[index]++;
		}
	}
	// check the first 3 stdev
	double expect68 = stdev_count[0] / n; // expect ~68%
	double expect95 = (stdev_count[0] + stdev_count[1]) / n; // expect ~95%
	double expect99 = (stdev_count[0] + stdev_count[1] + stdev_count[2]) / n; // expect ~99.7%

	double err1 = std::abs(0.68 - expect68);
	double err2 = std::abs(0.95 - expect95);
	double err3 = std::abs(0.997 - expect99);

	EXPECT_GT(ERR_THRESH, err1);
	EXPECT_GT(ERR_THRESH, err2);
	EXPECT_GT(ERR_THRESH, err3);
}


TEST_F(BBIND, Matmul_A032)
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
	std::vector<int64_t> output(nout);

	std::string opname = "matmul";
	ASSERT_TRUE(nnet::has_ele(opname));
	nnet::VTFUNC_F bifunc = nnet::ebind(opname);

	bifunc(nnet::INT64, nnet::VARR_T{(void*) &output[0], outshape}, {
		nnet::CVAR_T{(const void*) &argument0[0], shape0},
		nnet::CVAR_T{(const void*) &argument1[0], shape1},
	});

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


#endif /* DISABLE_BBIND_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */