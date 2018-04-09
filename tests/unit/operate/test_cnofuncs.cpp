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
	EXPECT_SHAPEQ(scalshape, ress);
	EXPECT_SHAPEQ(outshape, resscs);

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
	std::vector<size_t> clist = random_def_shape(this);
	size_t rank = clist.size();
	std::vector<uint64_t> perm(rank);
	std::iota(perm.begin(), perm.end(), 0);
	std::shuffle(perm.begin(), perm.end(),
		testify::get_generator());

	nnet::tensorshape shape = clist;
	std::vector<size_t> permlist(rank);
	for (size_t i = 0; i < rank; ++i)
	{
		permlist[i] = clist[perm[i]];
	}
	nnet::tensorshape permshape = permlist;
	std::reverse(clist.begin(), clist.end());
	nnet::tensorshape outshape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);
	nnet::varptr permleaf = nnet::constant::get<uint64_t>(perm, 
		nnet::tensorshape(std::vector<size_t>{rank}));

	nnet::varptr res = nnet::transpose(leaf);
	nnet::varptr resperm = nnet::transpose(leaf, permleaf);

	nnet::tensor* ten = res->get_tensor();
	nnet::tensor* tenperm = resperm->get_tensor();

	nnet::tensorshape ress = ten->get_shape();
	nnet::tensorshape resperms = tenperm->get_shape();
	EXPECT_SHAPEQ(outshape, ress);
	EXPECT_SHAPEQ(permshape, resperms);

	std::vector<double> data = nnet::expose<double>(res);
	std::vector<double> dataperm = nnet::expose<double>(tenperm);

	// test behavior B000
	nnet::varptr res2 = nnet::transpose(leaf);
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::transpose(nul).get());
	EXPECT_EQ(nullptr, nnet::transpose(nul, permleaf).get());
	EXPECT_EQ(res.get(), nnet::transpose(leaf, nul).get()); // invariant: behavior B000 is correct
	EXPECT_EQ(nullptr, nnet::transpose(nul, nul).get());

	// test behavior B1xx
	std::vector<size_t> coord;
	std::vector<size_t> tmp_coord;
	for (size_t i = 0; i < n; ++i)
	{
		tmp_coord = coord = permshape.coord_from_idx(i);
		for (size_t j = 0; j < perm.size(); ++j)
		{
			coord[perm[j]] = tmp_coord[j];
		}
		size_t permidx = shape.flat_idx(coord);
		EXPECT_EQ(argument[permidx], dataperm[i]);

		coord = outshape.coord_from_idx(i);
		std::reverse(coord.begin(), coord.end());
		size_t idx = shape.flat_idx(coord);
		EXPECT_EQ(argument[idx], data[i]);
	}

	nnet::varptr opres = nnet::run_opcode({leaf}, TRANSPOSE);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
}


TEST_F(CNOFUNCS, Flip_B0xxAndB144)
{
	std::vector<size_t> clist = random_def_shape(this);
	uint64_t argidx = get_int(1, "argidx", {0, clist.size() - 1})[0];

	nnet::tensorshape shape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);
	nnet::varptr dimleaf = nnet::constant::get<uint64_t>({argidx}, 
		nnet::tensorshape(std::vector<size_t>{1}));

	nnet::varptr res = nnet::flip(leaf, dimleaf);
	nnet::tensor* ten = res->get_tensor();
	nnet::tensorshape ress = ten->get_shape();
	EXPECT_SHAPEQ(shape, ress);

	std::vector<double> data = nnet::expose<double>(res);

	// test behavior B000
	nnet::varptr res2 = nnet::flip(leaf, dimleaf);
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::flip(nul, dimleaf).get());
	EXPECT_EQ(nullptr, nnet::flip(leaf, nul).get());
	EXPECT_EQ(nullptr, nnet::flip(nul, nul).get());

	// test behavior B1xx
	std::vector<size_t> coord;
	for (size_t i = 0; i < n; ++i)
	{
		coord = shape.coord_from_idx(i);
		coord[argidx] = clist[argidx] - coord[argidx] - 1;
		size_t idx = shape.flat_idx(coord);
		EXPECT_EQ(argument[idx], data[i]);
	}

	nnet::varptr opres = nnet::run_opcode({leaf, dimleaf}, FLIP);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
}


TEST_F(CNOFUNCS, ExpandB0xxAndB145)
{
	std::vector<size_t> clist = random_def_shape(this);
	uint64_t argidx = get_int(1, "argidx", {0, clist.size()})[0];
	uint64_t mult = get_int(1, "mult", {1, 6})[0];

	nnet::tensorshape shape = clist;
	clist.insert(clist.begin() + argidx, mult);
	nnet::tensorshape outshape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);
	nnet::varptr multleaf = nnet::constant::get<double>({(double) mult}, 
		nnet::tensorshape(std::vector<size_t>{1})); // todo: make this uint64_t once shape_func uses uint64_t
	nnet::varptr dimleaf = nnet::constant::get<uint64_t>({argidx}, 
		nnet::tensorshape(std::vector<size_t>{1}));

	nnet::varptr res = nnet::expand(leaf, multleaf, dimleaf);
	nnet::tensor* ten = res->get_tensor();
	nnet::tensorshape ress = ten->get_shape();
	EXPECT_SHAPEQ(outshape, ress);

	std::vector<double> data = nnet::expose<double>(res);

	// test behavior B000
	nnet::varptr res2 = nnet::expand(leaf, multleaf, dimleaf);
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::expand(nul, multleaf, dimleaf).get());
	EXPECT_EQ(nullptr, nnet::expand(leaf, nul, dimleaf).get());
	EXPECT_EQ(nullptr, nnet::expand(leaf, multleaf, nul).get());
	EXPECT_EQ(nullptr, nnet::expand(nul, nul, dimleaf).get());
	EXPECT_EQ(nullptr, nnet::expand(nul, multleaf, nul).get());
	EXPECT_EQ(nullptr, nnet::expand(leaf, nul, nul).get());
	EXPECT_EQ(nullptr, nnet::expand(nul, nul, nul).get());

	// test behavior B1xx
	std::vector<size_t> coord;
	for (size_t i = 0; i < n; ++i)
	{
		coord = shape.coord_from_idx(i);
		coord.insert(coord.begin() + argidx, 0);
		for (size_t j = 0; j < mult; ++j)
		{
			coord[argidx] = j;
			size_t idx = outshape.flat_idx(coord);
			EXPECT_EQ(argument[i], data[idx]);
		}
	}

	nnet::varptr opres = nnet::run_opcode({leaf, multleaf, dimleaf}, EXPAND);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
}


TEST_F(CNOFUNCS, Nelems_B0xxAndB146)
{
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);

	nnet::varptr res = nnet::n_elems(leaf);
	nnet::tensor* ten = res->get_tensor();
	nnet::tensorshape ress = ten->get_shape();
	nnet::tensorshape outshape(std::vector<size_t>{1});
	EXPECT_SHAPEQ(outshape, ress);

	std::vector<double> data = nnet::expose<double>(res);

	// test behavior B000
	nnet::varptr res2 = nnet::n_elems(leaf);
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	EXPECT_EQ(nullptr, nnet::n_elems(nnet::varptr()).get());

	// test behavior B1xx
	EXPECT_EQ(n, data[0]);

	nnet::varptr opres = nnet::run_opcode({leaf}, N_ELEMS);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
}


TEST_F(CNOFUNCS, Ndims_B0xxAndB147)
{
	std::vector<size_t> clist = random_def_shape(this);
	uint64_t argidx = get_int(1, "argidx", {0, clist.size() - 1})[0];
	nnet::tensorshape shape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);

	nnet::varptr res = nnet::n_dimension(leaf, argidx);
	nnet::tensor* ten = res->get_tensor();
	nnet::tensorshape ress = ten->get_shape();
	nnet::tensorshape outshape(std::vector<size_t>{1});
	EXPECT_SHAPEQ(outshape, ress);

	std::vector<double> data = nnet::expose<double>(res);

	// test behavior B000
	nnet::varptr res2 = nnet::n_dimension(leaf, argidx);
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::n_dimension(nul, argidx).get());
	EXPECT_EQ(nnet::n_dimension(leaf, 0).get(), nnet::n_dimension(leaf, nul).get());
	EXPECT_EQ(nullptr, nnet::n_dimension(nul, nul).get());

	// test behavior B1xx
	EXPECT_EQ(clist[argidx], data[0]);

	nnet::varptr opres = nnet::run_opcode({leaf, nnet::constant::get<uint64_t>(argidx)}, N_DIMS);
	EXPECT_EQ(res.get(), opres.get()); // invariant: behavior B000 is successful
}


TEST_F(CNOFUNCS, Clip_B0xxAndB148)
{
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);
	double mi = get_double(1, "min", {-1, 0})[0];
	double ma = get_double(1, "max", {0, 1})[0];

	nnet::varptr res = nnet::clip(leaf, mi, ma);
	nnet::tensor* ten = res->get_tensor();
	nnet::tensorshape ress = ten->get_shape();
	EXPECT_SHAPEQ(shape, ress);

	std::vector<double> data = nnet::expose<double>(res);

	// test behavior B000
	nnet::varptr res2 = nnet::clip(leaf, mi, ma);
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	nnet::varptr nul;
	nnet::varptr leafmi = nnet::constant::get(mi);
	nnet::varptr leafma = nnet::constant::get(ma);
	EXPECT_EQ(nullptr, nnet::clip(nul, leafmi, leafma).get());
	EXPECT_EQ(nullptr, nnet::clip(leaf, nul, leafma).get());
	EXPECT_EQ(nullptr, nnet::clip(leaf, leafmi, nul).get());
	EXPECT_EQ(nullptr, nnet::clip(nul, nul, leafma).get());
	EXPECT_EQ(nullptr, nnet::clip(nul, leafmi, nul).get());
	EXPECT_EQ(nullptr, nnet::clip(leaf, nul, nul).get());
	EXPECT_EQ(nullptr, nnet::clip(nul, nul, nul).get());
	// test behavior B1xx
	for (size_t i = 0; i < n; ++i)
	{
		double a = argument[i];
		if (a < mi)
		{
			a = mi;
		}
		else if (a > ma)
		{
			a = ma;
		}
		EXPECT_EQ(a, data[i]);
	}
}


TEST_F(CNOFUNCS, ClipNorm_B0xxAndB149)
{
	std::vector<size_t> clist = random_def_shape(this);
	nnet::tensorshape shape = clist;
	size_t n = shape.n_elems();
	std::vector<double> argument = get_double(n, "argument", {-1, 1});
	nnet::varptr leaf = nnet::constant::get<double>(argument, shape);
	double l2norm = std::sqrt(std::accumulate(argument.begin(), argument.end(), (double) 0,
	[](double accum, double elem)
	{
		return accum + elem * elem;
	}));
	double cap = get_double(1, "cap", {l2norm / 2, 3 * l2norm / 2})[0];

	nnet::varptr res = nnet::clip_norm(leaf, cap);
	nnet::tensor* ten = res->get_tensor();
	nnet::tensorshape ress = ten->get_shape();
	EXPECT_SHAPEQ(shape, ress);

	std::vector<double> data = nnet::expose<double>(res);

	// test behavior B000
	nnet::varptr res2 = nnet::clip_norm(leaf, cap);
	EXPECT_EQ(res.get(), res2.get());

	// test behavior B001
	nnet::varptr nul;
	EXPECT_EQ(nullptr, nnet::clip_norm(nul, cap).get());
	EXPECT_EQ(nullptr, nnet::clip_norm(leaf, nul).get());
	EXPECT_EQ(nullptr, nnet::clip_norm(nul, nul).get());
	// test behavior B1xx
	if (cap < l2norm)
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(argument[i] * cap / l2norm, data[i]);
		}
	}
	else
	{
		for (size_t i = 0; i < n; ++i)
		{
			EXPECT_EQ(argument[i], data[i]);
		}
	}
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
