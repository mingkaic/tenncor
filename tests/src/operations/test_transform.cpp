//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATION_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "include/graph/leaf/variable.hpp"
#include "include/operations/operations.hpp"

#include "tests/include/utils/util_test.h"
#include "tests/include/utils/fuzz.h"


#ifndef DISABLE_TRANSFORM_TEST


class TRANSFORM : public FUZZ::fuzz_test {};


using namespace nnet;


using SHAPE_CHANGE = std::function<tensorshape(tensorshape)>;
using DATA_CHANGE = std::function<std::vector<double>(std::vector<double>,tensorshape,tensorshape)>;

template <typename T>
using PARAM_EVAL = std::function<T(tensorshape)>;
template <typename T>
using UNARY_VAR = std::function<varptr(varptr,T)>;


template <typename T>
T no_param (tensorshape) { return (T)0; }


tensorshape as_is (tensorshape in) { return in; }


std::vector<double> onescalar (std::vector<double>, tensorshape, tensorshape)
{
	return std::vector<double>(1, 1);
}


template <typename T=double>
static void unaryTransTest (FUZZ::fuzz_test* fuzzer,
	std::pair<int,int> ranklimit, UNARY_VAR<T> func,
	DATA_CHANGE expect_transfer, SHAPE_CHANGE expect_shape,
	optional<DATA_CHANGE> grad_transfer, optional<SHAPE_CHANGE> grad_shape,
	PARAM_EVAL<T> paramer = no_param<T>)
{
	tensorshape shape = random_def_shape(fuzzer, ranklimit.first, ranklimit.second);
	rand_uniform rinit(2, 12);
	variable var(shape, rinit, nnet::DOUBLE, "unar_var");
	var.initialize();
	{
		const tensor* vartens = var.eval();
		ASSERT_NE(nullptr, vartens);
		ASSERT_TRUE(vartens->is_alloc());
	}
	varptr res = func(varptr(&var), paramer(shape));
	{
		const tensor* restens = res->eval();
		ASSERT_NE(nullptr, restens);
		ASSERT_TRUE(restens->is_alloc());
	}

	tensorshape expectoshape = expect_shape(shape);
	std::vector<double> varout = expose<double>(&var);
	std::vector<double> expectout = expect_transfer(varout, shape, expectoshape);
	nnet::inode* vgrad = var.derive(&var);
	{
		const tensor* vgradtens = vgrad->eval();
		ASSERT_NE(nullptr, vgradtens);
		ASSERT_TRUE(vgradtens->is_alloc());
	}

	// test forward
	tensorshape outshape = res->get_shape();
	std::vector<double> rout = expose<double>(res);
	EXPECT_TRUE(tensorshape_equal(expectoshape, outshape));
	ASSERT_EQ(expectout.size(), rout.size());
	for (size_t i = 0, n = rout.size(); i < n; i++)
	{
		EXPECT_EQ(expectout[i], rout[i]);
	}

	// test derivative
	if ((bool) grad_transfer && (bool) grad_shape)
	{
		tensorshape gradoshape = (*grad_shape)(var.derive(&var)->get_shape());
		std::vector<double> gradout =
			(*grad_transfer)(expose<double>(vgrad), vgrad->get_shape(), gradoshape);
		const tensor_double* backt =
			dynamic_cast<const tensor_double*>(res->derive(&var)->eval());
		tensorshape outgshape = backt->get_shape();
		std::vector<double> rgout = backt->expose();
		EXPECT_TRUE(tensorshape_equal(gradoshape, outgshape));
		ASSERT_EQ(gradout.size(), rgout.size());
		for (size_t i = 0, n = rgout.size(); i < n; i++)
		{
			EXPECT_EQ(gradout[i], rgout[i]);
		}
	}
	else
	{
		EXPECT_THROW(res->derive(&var), std::exception);
	}

	// Behavior B000
	EXPECT_EQ(nullptr, func(varptr(nullptr), paramer(shape)));
}


TEST_F(TRANSFORM, L2norm_)
{
	// todo: add to behavior.txt
	DATA_CHANGE transfer =
	[](std::vector<double> in, tensorshape inshape,
		tensorshape outshape) -> std::vector<double>
	{
		size_t n = inshape.n_elems();
		double out = 0;
		for (size_t i = 0; i < n; i++)
		{
			out += in[i] * in[i];
		}
		return std::vector<double>{std::sqrt(out)};
	};
	SHAPE_CHANGE shape =
	[](tensorshape in) -> tensorshape
	{
		return std::vector<size_t>{1};
	};
	unaryTransTest<double>(this, {1, 2},
	[](varptr in,double) { return nnet::reduce_l2norm(in); },
	transfer, shape, transfer, shape);
}


TEST_F(TRANSFORM, Transpose_B001)
{
	DATA_CHANGE transfer =
	[](std::vector<double> in, tensorshape inshape,
		tensorshape outshape) -> std::vector<double>
	{
		if (1 == inshape.rank())
		{
			return in;
		}
		std::vector<size_t> slist = inshape.as_list();
		std::vector<double> out(in.size(), 0);
		for (size_t i = 0, n = in.size(); i < n; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			size_t j = outshape.flat_idx({incoord[1], incoord[0]});
			out[j] = in[i];
		}
		return out;
	};
	SHAPE_CHANGE shape =
	[](tensorshape in) -> tensorshape
	{
		if (1 == in.rank())
		{
			return std::vector<size_t>{1, in.as_list()[0]};
		}
		std::vector<size_t> slist = in.as_list();
		return std::vector<size_t>{slist[1], slist[0]};
	};
	unaryTransTest<double>(this, {1, 2},
	[](varptr in,double) { return nnet::transpose(in); },
	transfer, shape, transfer, shape);
}


TEST_F(TRANSFORM, Fit_B002)
{
	tensorshape realshape = random_def_shape(this);
	rand_uniform rinit(2, 12);
	variable shapeholder(realshape, rinit, nnet::DOUBLE, "shapeholder");
	shapeholder.initialize();

	PARAM_EVAL<const varptr > fitparam =
	[&shapeholder](tensorshape) -> const varptr
	{
		return varptr(&shapeholder);
	};
	DATA_CHANGE transfer =
	[](std::vector<double> in, tensorshape inshape, tensorshape outshape) -> std::vector<double>
	{
		size_t n = outshape.n_elems();
		std::vector<double> out(n, 0);
		std::vector<size_t> outlist = outshape.as_list();
		for (size_t i = 0, m = in.size(); i < m; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			bool b = true;
			for (size_t j = 0, o = incoord.size(); j < o && b; j++)
			{
				if (j >= outlist.size())
				{
					b = incoord[j] == 0;
				}
				else
				{
					b = incoord[j] < outlist[j];
				}
			}
			if (b)
			{
				size_t outidx = outshape.flat_idx(incoord);
				out[outidx] = in[i];
			}
		}
		return out;
	};
	SHAPE_CHANGE shape = [&realshape](tensorshape) { return realshape; };

	optional<DATA_CHANGE> gradtransfer = (DATA_CHANGE) onescalar;
	optional<SHAPE_CHANGE> gradshape = (SHAPE_CHANGE) as_is;

	unaryTransTest<const varptr>(this, {2, 13},
	[](varptr in, varptr watch) { return nnet::fit(in, watch); },
	transfer, shape, gradtransfer, gradshape, fitparam);
}


TEST_F(TRANSFORM, Extend_B003To004)
{
	// B004
	size_t extend_index;
	size_t multiplier;
	PARAM_EVAL<std::pair<size_t,size_t> > extendparam =
	[this, &extend_index, &multiplier](tensorshape shape) -> std::pair<size_t,size_t>
	{
		size_t srank = shape.rank();
		extend_index = get_int(1, "extend_index", {0, srank-1})[0];
		multiplier = get_int(1, "multiplier", {2, 5})[0];
		return {extend_index, multiplier};
	};
	DATA_CHANGE transfer =
	[&extend_index, &multiplier](std::vector<double> in, tensorshape inshape, tensorshape) -> std::vector<double>
	{
		std::vector<size_t> invec = inshape.as_list();
		std::vector<double> out;
		size_t baselen = 1;
		for (size_t i = 0; i <= extend_index; i++)
		{
			baselen *= invec[i];
		}
		auto it = in.begin();
		auto et = in.end();
		while (it != et)
		{
			for (size_t i = 0; i < multiplier; i++)
			{
				out.insert(out.end(), it, it+baselen);
			}
			it += baselen;
		}
		return out;
	};
	SHAPE_CHANGE shape =
	[&extend_index, &multiplier](tensorshape inshape) -> tensorshape
	{
		std::vector<size_t> out = inshape.as_list();
		out[extend_index] *= multiplier;
		return out;
	};

	optional<DATA_CHANGE> gradtransfer = (DATA_CHANGE) onescalar;
	optional<SHAPE_CHANGE> gradshape = (SHAPE_CHANGE) as_is;

	unaryTransTest<std::pair<size_t,size_t> >(this, {2, 13},
	[](varptr in, std::pair<size_t,size_t> idxnmult)
	{
		size_t index = idxnmult.first;
		size_t multiplier = idxnmult.second;
		return extend(in, index, multiplier);
	},
	transfer, shape, gradtransfer, gradshape, extendparam);
	// B005
	tensorshape rshape = random_def_shape(this, 2, 13);
	rand_uniform rinit(2, 12);
	variable var(rshape, rinit, nnet::DOUBLE, "unar_var");
	varptr zaro = extend(varptr(&var), extend_index, 0);
	var.initialize();
	const tensor_double* ztens = dynamic_cast<const tensor_double*>(zaro->eval());
	EXPECT_EQ((size_t) 1, ztens->get_shape().n_elems());
	std::vector<double> zvec = ztens->expose();
	ASSERT_EQ((size_t) 1, zvec.size());
	EXPECT_EQ(0.0, zvec[0]);
	varptr same = extend(varptr(&var), extend_index, 1);
	EXPECT_EQ(&var, same.get());
}


void compressTestCommon (FUZZ::fuzz_test* fuzzer,
	UNARY_VAR<size_t> trans, BI_TRANS<double> setCompression,
	std::function<void(size_t,std::vector<double>&)> additionalWork =
	[](size_t,std::vector<double>&) {},
	DATA_CHANGE gradtransfer = onescalar)
{
	size_t compress_index;
	PARAM_EVAL<size_t> compressparam =
	[fuzzer, &compress_index](tensorshape shape) -> size_t
	{
		size_t srank = shape.rank();
		compress_index = fuzzer->get_int(1, "compress_index", {0, srank-1})[0];
		return compress_index;
	};
	DATA_CHANGE transfer =
	[&compress_index, setCompression, additionalWork](std::vector<double> in,
		tensorshape inshape, tensorshape outshape) -> std::vector<double>
	{
		if (compress_index >= inshape.rank()) return in;
		std::vector<double> out(outshape.n_elems(), 0);
		for (size_t i = 0, m = in.size(); i < m; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			if (compress_index == 0)
			{
				incoord = std::vector<size_t>(incoord.begin() + 1, incoord.end());
			}
			else if (compress_index == incoord.size() - 1)
			{
				incoord.pop_back();
			}
			else
			{
				incoord[compress_index] = 0;
			}

			size_t outidx = outshape.flat_idx(incoord);
			out[outidx] = setCompression(out[outidx], in[i]);
		}
		additionalWork(inshape.as_list()[compress_index], out);
		return out;
	};
	SHAPE_CHANGE shape =
	[&compress_index](tensorshape inshape) -> tensorshape
	{
		std::vector<size_t> out = inshape.as_list();
		if (compress_index >= out.size()) return inshape;
		if (compress_index == 0)
		{
			out = std::vector<size_t>(out.begin()+1, out.end());
		}
		else if (out.size()-1 == (unsigned) compress_index)
		{
			out.pop_back();
		}
		else
		{
			out[compress_index] = 1;
		}
		return out;
	};

	unaryTransTest<size_t>(fuzzer, {2, 13}, trans,
	transfer, shape, gradtransfer, (SHAPE_CHANGE) as_is, compressparam);
}


TEST_F(TRANSFORM, Compress_B005)
{
	BI_TRANS<double> compression =
	[](double a, double b) -> double
	{
		return a + b;
	};

	compressTestCommon(this, [&compression](varptr in, size_t compidx)
	{
		return compress(in, compression, compidx);
	}, compression);
}


TEST_F(TRANSFORM, CompressScalar_B006)
{
	tensorshape shape = random_def_shape(this, 2, 13);
	rand_uniform rinit(2, 12);
	variable var(shape, rinit, nnet::DOUBLE, "unar_var");
	var.initialize();

	std::vector<BI_TRANS<double>> comps = {
		[](double a, double b)
		{
			return a + b;
		},
		[](double a, double b)
		{
			return std::max(a, b);
		},
		[](double a, double b)
		{
			return std::min(a, b);
		}
	};
	BI_TRANS<double> comp = comps[get_int(1, "compIdx", {0, comps.size() - 1})[0]];

	varptr scal = compress(varptr(&var), comp, optional<size_t>());

	std::vector<double> raw = expose<double>(&var);
	double real_val = std::accumulate(raw.begin() + 1, raw.end(), raw[0], comp);
	EXPECT_EQ(real_val, expose<double>(scal)[0]);
}


TEST_F(TRANSFORM, ReduceMax_)
{
	// todo: add to behavior.txt
	BI_TRANS<double> compression =
	[](double a, double b) -> double
	{
		return std::max(a, b);
	};

	compressTestCommon(this, [](varptr in, size_t compidx)
	{ return reduce_max(in, compidx); }, compression);
}


TEST_F(TRANSFORM, ReduceSum_)
{
	// todo: add to behavior.txt
	BI_TRANS<double> compression =
	[](double a, double b) -> double
	{
		return a + b;
	};

	compressTestCommon(this, [](varptr in, size_t compidx)
	{ return reduce_sum(in, compidx); }, compression);
}


TEST_F(TRANSFORM, ReduceMean_)
{
	// todo: add to behavior.txt
	BI_TRANS<double> compression =
	[](double a, double b) -> double
	{
		return a + b;
	};
	size_t nchange;

	compressTestCommon(this, [](varptr in, size_t compidx)
	{ return reduce_mean(in, compidx); }, compression,
	[&nchange](size_t compn, std::vector<double>& out)
	{
		for (double& o : out)
		{
			o /= compn;
		}
		nchange = compn;
	}, [&nchange](std::vector<double> data, tensorshape, tensorshape) -> std::vector<double>
	{
		return std::vector<double>(1, 1.0 / nchange);
	});
}


void argcompTestCommon (FUZZ::fuzz_test* fuzzer,
	UNARY_VAR<size_t> trans, REDUCE<double> setArgcomp)
{
	size_t arg_index;
	REDUCE<double> search =
	[](std::vector<double> data) -> double
	{
		return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
	};

	PARAM_EVAL<size_t> argcompressparam =
	[fuzzer, &arg_index](tensorshape shape) -> size_t
	{
		size_t srank = shape.rank();
		arg_index = fuzzer->get_int(1, "arg_index", {0, srank-1})[0];
		return arg_index;
	};
	DATA_CHANGE transfer =
	[&arg_index, setArgcomp](std::vector<double> in, tensorshape inshape,
		tensorshape outshape) -> std::vector<double>
	{
		assert(arg_index < inshape.rank());
		std::vector<double> out(outshape.n_elems(), 0);
		std::vector<std::vector<double> > out_searches(outshape.n_elems(), std::vector<double>{});
		for (size_t i = 0, m = in.size(); i < m; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			if (arg_index == 0)
			{
				incoord = std::vector<size_t>(incoord.begin() + 1, incoord.end());
			}
			else if (arg_index == incoord.size() - 1)
			{
				incoord.pop_back();
			}
			else
			{
				incoord[arg_index] = 0;
			}
			std::vector<double> vecs;
			size_t outidx = outshape.flat_idx(incoord);
			out_searches[outidx].push_back(in[i]);
		}
		std::transform(out_searches.begin(), out_searches.end(), out.begin(),
		[setArgcomp](std::vector<double>& vec)
		{
			return setArgcomp(vec);
		});
		return out;
	};
	SHAPE_CHANGE shape =
	[&arg_index](tensorshape inshape) -> tensorshape
	{
		std::vector<size_t> out = inshape.as_list();
		assert(arg_index < out.size());
		if (arg_index == 0)
		{
			out = std::vector<size_t>(out.begin()+1, out.end());
		}
		else if (out.size()-1 == (unsigned) arg_index)
		{
			out.pop_back();
		}
		else
		{
			out[arg_index] = 1;
		}
		return out;
	};

	optional<DATA_CHANGE> gradtransfer;
	optional<SHAPE_CHANGE> gradshape;

	unaryTransTest<size_t>(fuzzer, {3, 13}, trans,
	transfer, shape, gradtransfer, gradshape, argcompressparam);
}


TEST_F(TRANSFORM, ArgCompress_B008To009)
{
	REDUCE<double> search =
	[](std::vector<double> data) -> double
	{
		return std::distance(data.begin(), std::min_element(data.begin(), data.end()));
	};
	argcompTestCommon(this, [&search](varptr in, size_t arg_index)
	{ return arg_compress(in, search, arg_index); }, search);
}


TEST_F(TRANSFORM, ArgCompressScalar_B010)
{
	tensorshape shape = random_def_shape(this, 2, 13);
	rand_uniform rinit(2, 12);
	variable var(shape, rinit, nnet::DOUBLE, "unar_var");
	var.initialize();

	std::vector<REDUCE<double>> comps = {
		[](std::vector<double> data)
		{
			double accum = 0;
			size_t idx = 0;
			for (size_t i = 0, n = data.size(); i < n; i++)
			{
				if (accum < data[i])
				{
					accum = data[i];
					idx = i;
				}
			}
			return (double) idx;
		},
		[](std::vector<double> data)
		{
			double accum = 0;
			size_t idx = 0;
			for (size_t i = 0, n = data.size(); i < n; i++)
			{
				if (accum < data[i])
				{
					accum = data[i];
					idx = i;
				}
			}
			return (double) idx;
		}
	};
	REDUCE<double> comp = comps[get_int(1, "compIdx", {0, comps.size() - 1})[0]];

	varptr scal = arg_compress(varptr(&var), comp, optional<size_t>());

	std::vector<double> raw = expose<double>(&var);
	double real_idx = comp(raw);
	EXPECT_EQ(real_idx, expose<double>(scal)[0]);
}


TEST_F(TRANSFORM, ArgMax_)
{
	// todo: add to behavior.txt
	REDUCE<double> search =
	[](std::vector<double> data) -> double
	{
		return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
	};
	argcompTestCommon(this, [](varptr in, size_t arg_index)
	{ return arg_max(in, arg_index); }, search);
}


TEST_F(TRANSFORM, Flip_B012)
{
	tensorshape shape = random_def_shape(this);
	std::vector<size_t> shapelist = shape.as_list();
	size_t inn = shape.n_elems();
	rand_uniform rinit(2, 12);

	size_t flipdim = get_int(1, "flipdim", {0, shape.rank() - 1})[0];

	variable var(shape, rinit, nnet::DOUBLE, "unar_var");
	varptr res = flip(varptr(&var), std::vector<size_t>{ flipdim });

	// Behavior A000
	EXPECT_EQ(nullptr, flip(nullptr, std::vector<size_t>{0}));

	// initialize
	var.initialize();
	std::vector<double> indata = expose<double>(&var);

	// compare data, shape must be equivalent, since we're testing elementary operations (Behavior A001)
	const tensor_double* rawtens = dynamic_cast<const tensor_double*>(res->eval());
	std::vector<double> rawf = rawtens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, rawtens->get_shape()));
	ASSERT_EQ(rawf.size(), inn);
	for (size_t i = 0; i < inn; i++)
	{
		std::vector<size_t> coord = shape.coordinate_from_idx(i);
		coord[flipdim] = shapelist[flipdim] - coord[flipdim] - 1;
		double out = rawf[i];
		double in = indata[shape.flat_idx(coord)];
		EXPECT_EQ(in, out);
	}

	// todo: (test) flip back prop
}


TEST_F(TRANSFORM, DISABLED_CrossCorr2d_)
{
	// todo: implement + add to behavior.txt
}


TEST_F(TRANSFORM, DISABLED_Conv2d_)
{
	// todo: implement + add to behavior.txt
}


#endif /* DISABLE_TRANSFORM_TEST */


#endif /* DISABLE_OPERATION_MODULE_TESTS */
