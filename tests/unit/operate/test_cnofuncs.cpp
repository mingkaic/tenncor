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


#ifndef DISABLE_CNOFUNCS_TEST // compound node functions


static const double ERR_THRESH = 0.08; // 8% error


struct CNOFUNCS : public testutils::fuzz_test {};


using namespace testutils;


using VARFUNC = std::function<nnet::varptr(std::vector<nnet::varptr>)>;


using VARSCALFUNC = std::function<nnet::varptr(nnet::varptr,uint64_t)>;


using AGGS = std::function<void(double,std::vector<double>,std::vector<double>)>;


static void unaryAggTest (testutils::fuzz_test* fuzzer, 
	OPCODE opcode, VARFUNC op, VARSCALFUNC opscalar, 
	AGGS agg, std::pair<double,double> limits = {-1, 1})
{
	std::vector<size_t> clist = random_def_shape(fuzzer);
	uint64_t argidx = fuzzer->get_int(1, "argidx", {0, clist.size() - 1})[0];
	nnet::tensorshape shape = clist;
	if (1 == clist.size())
	{
		clist[0] = 1;
	}
	else
	{
		clist.erase(clist.begin() + argidx);
	}
	nnet::tensorshape scalshape = std::vector<size_t>{1};
	nnet::tensorshape outshape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = fuzzer->get_double(n, "argument", limits);
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);

	nnet::varptr res = op({leaf});
	nnet::varptr resscalar = opscalar(leaf, argidx);

	nnet::tensor* ten = res->get_tensor();
	nnet::tensor* tenscalar = resscalar->get_tensor();

	nnet::tensorshape ress = ten->get_shape();
	nnet::tensorshape resscs = tenscalar->get_shape();
	EXPECT_TRUE(tensorshape_equal(scalshape, ress)) <<
		sprintf("expect %p, got %p", &scalshape, &ress);
	EXPECT_TRUE(tensorshape_equal(outshape, resscs)) <<
		sprintf("expect %p, got %p", &outshape, &resscs);

	std::vector<double> data = nnet::expose<double>(res);
	std::vector<double> datascalar = nnet::expose<double>(resscalar);

	// test behavior B000
	nnet::varptr res2 = op({leaf});
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, op({nul}).get());
	EXPECT_EQ(nullptr, opscalar(nul, argidx).get());
	EXPECT_EQ(opscalar(leaf, 0).get(), op({leaf, nul}).get()); // invariant: behavior B000 is correct
	EXPECT_EQ(nullptr, op({nul, nul}).get());

	// test behavior B1xx
	agg(data[0], argument, argument);

	size_t m = outshape.n_elems();
	std::vector<std::vector<double> > out_searches(m, std::vector<double>{});
	for (size_t i = 0; i < n; i++)
	{
		std::vector<size_t> incoord = shape.coord_from_idx(i);
		if (1 == incoord.size())
		{
			incoord[0] = 1;
		}
		else
		{
			incoord.erase(incoord.begin() + argidx);
		}
		size_t outidx = outshape.flat_idx(incoord);
		out_searches[outidx].push_back(argument[i]);
	}
	for (size_t i = 0; i < m; ++i)
	{
		auto& vec = out_searches[i];
		agg(datascalar[i], vec, argument);
	}

	if (opcode < _OP_SENTINEL)
	{
		nnet::varptr opres = nnet::run_opcode({leaf}, opcode);
		EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
	}
}


TEST_F(CNOFUNCS, Argmax_B0xxAndB140)
{
	unaryAggTest(this, ARGMAX,
	[](std::vector<nnet::varptr> vars)
	{
		if (vars.size() > 1)
		{
			return nnet::arg_max(vars[0], vars[1]);
		}
		return nnet::arg_max(vars[0]);
	},
	[](nnet::varptr a, uint64_t scalar)
	{
		return nnet::arg_max(a, scalar);
	},
	[](double got, std::vector<double> expecting, std::vector<double> input)
	{
		EXPECT_EQ(*std::max_element(expecting.begin(), expecting.end()), input[(size_t) got]);
	});
}


TEST_F(CNOFUNCS, Rmax_B0xxAndB141)
{
	unaryAggTest(this, RMAX,
	[](std::vector<nnet::varptr> vars)
	{
		if (vars.size() > 1)
		{
			return nnet::reduce_max(vars[0], vars[1]);
		}
		return nnet::reduce_max(vars[0]);
	},
	[](nnet::varptr a, uint64_t scalar)
	{
		return nnet::reduce_max(a, scalar);
	},
	[](double got, std::vector<double> expecting, std::vector<double>)
	{
		EXPECT_EQ(*std::max_element(expecting.begin(), expecting.end()), got);
	});
}


TEST_F(CNOFUNCS, Rsum_B0xxAndB142)
{
	unaryAggTest(this, RSUM,
	[](std::vector<nnet::varptr> vars)
	{
		if (vars.size() > 1)
		{
			return nnet::reduce_sum(vars[0], vars[1]);
		}
		return nnet::reduce_sum(vars[0]);
	},
	[](nnet::varptr a, uint64_t scalar)
	{
		return nnet::reduce_sum(a, scalar);
	},
	[](double got, std::vector<double> expecting, std::vector<double>)
	{
		EXPECT_EQ(std::accumulate(expecting.begin(), expecting.end(), (double) 0), got);
	});
}


TEST_F(CNOFUNCS, Transpose_B0xxAndB143)
{
}


TEST_F(CNOFUNCS, Flip_B0xxAndB144)
{
}


TEST_F(CNOFUNCS, ExpandB0xxAndB145)
{
}


TEST_F(CNOFUNCS, Nelems_B0xxAndB146)
{
}


TEST_F(CNOFUNCS, Ndims_B0xxAndB147)
{
}


TEST_F(CNOFUNCS, Clip_B0xxAndB148)
{
}


TEST_F(CNOFUNCS, ClipNorm_B0xxAndB149)
{
}


TEST_F(CNOFUNCS, Rmean_B0xxAndB150)
{
	unaryAggTest(this, _OP_SENTINEL,
	[](std::vector<nnet::varptr> vars)
	{
		if (vars.size() > 1)
		{
			return nnet::reduce_mean(vars[0], vars[1]);
		}
		return nnet::reduce_mean(vars[0]);
	},
	[](nnet::varptr a, uint64_t scalar)
	{
		return nnet::reduce_mean(a, scalar);
	},
	[](double got, std::vector<double> expecting, std::vector<double>)
	{
		EXPECT_EQ(std::accumulate(expecting.begin(), expecting.end(), (double) 0) / expecting.size(), got);
	});
}


TEST_F(CNOFUNCS, Rnorm_B0xxAndB151)
{
	unaryAggTest(this, _OP_SENTINEL,
	[](std::vector<nnet::varptr> vars)
	{
		if (vars.size() > 1)
		{
			return nnet::reduce_l2norm(vars[0], vars[1]);
		}
		return nnet::reduce_l2norm(vars[0]);
	},
	[](nnet::varptr a, uint64_t scalar)
	{
		return nnet::reduce_l2norm(a, scalar);
	},
	[](double got, std::vector<double> expecting, std::vector<double>)
	{
		double l2 = std::sqrt(std::accumulate(expecting.begin(), expecting.end(), (double) 0, 
		[](double acc, double d) -> double
		{
			return acc + (d * d);
		}));
		EXPECT_EQ(l2, got);
	});
}


using namespace testutils;


#endif /* DISABLE_CNOFUNCS_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
