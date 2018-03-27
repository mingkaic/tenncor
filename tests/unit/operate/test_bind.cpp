//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATE_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "include/graph/variable.hpp"
#include "include/operate/operations.hpp"

#include "tests/unit/include/utils/util_test.hpp"
#include "tests/unit/include/utils/fuzz.h"


#ifndef DISABLE_BIND_TEST


class ELEMENTARY : public FUZZ::fuzz_test {};


using namespace nnet;


using UNARY_SCALAR = std::function<double(double)>;
using UNARY_VAR = std::function<varptr(varptr)>;
using BINARY_SCALARS = std::function<double(double, double)>;
using BINARY_VARS = std::function<varptr(varptr, varptr)>;
using BINARY_VAR1 = std::function<varptr(varptr, double)>;
using BINARY_VAR2 = std::function<varptr(double, varptr)>;
using QUANARY_SCALARS = std::function<double(double, double, double, double)>;


static const double epi = std::numeric_limits<double>::epsilon();


// commonly used testing format
static void unaryElemTest (FUZZ::fuzz_test* fuzzer, UNARY_VAR func,
	UNARY_SCALAR expect_forward, BINARY_SCALARS expect_back,
	std::function<void(std::vector<double>&)> primer =
	[](std::vector<double>&) {})
{
	tensorshape shape = random_def_shape(fuzzer);
	size_t inn = shape.n_elems();
	rand_uniform rinit(2, 12);

	variable var(shape, rinit, nnet::DOUBLE, "unar_var");
	varptr res = func(varptr(&var));

	// Behavior A000
	EXPECT_EQ(nullptr, func(varptr(nullptr)));

	// initialize
	var.initialize();
	std::vector<double> indata = expose<double>(&var);

	// compare data, shape must be equivalent, since we're testing elementary operations (Behavior A002)
	const tensor_double* rawtens = dynamic_cast<const tensor_double*>(res->eval());
	std::vector<double> rawf = rawtens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, rawtens->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_EQ(rawf.size(), inn);
	primer(indata);
	for (size_t i = 0; i < inn; i++)
	{
		double rawd = rawf[i];
		double forwardd = expect_forward(indata[i]);
		// allow some error in case of rounding error
		double errf = std::abs(forwardd - rawd);
		EXPECT_GE(epi, errf);
	}

	varptr grad = res->derive(&var);
	const tensor_double* backtens = dynamic_cast<const tensor_double*>(grad->eval());
	std::vector<double> rawb = backtens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, backtens->get_shape()) || rawb.size() == 1) <<
		sprintf("expecting shape %p, got %p", &shape, );
	primer(indata);
	if (rawb.size() == 1)
	{
		double rawdb = rawb[0];
		double backd = expect_back(indata[0], 1);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawdb);
		EXPECT_GE(epi, errb);
	}
	else
	{
		for (size_t i = 0; i < inn; i++)
		{
			double rawdb = rawb[i];
			double backd = expect_back(indata[i], 1);
			// allow some error in case of rounding error
			double errb = std::abs(backd - rawdb);
			EXPECT_GE(epi, errb);
		}
	}

	// behavior A001
	// avoid negatives to prevent bad ops
	std::vector<double> constant_values = fuzzer->get_double(shape.n_elems(), "constant_values", {0, 17});
	nnet::constant* c = nnet::constant::get<double>(constant_values, shape);
	nnet::varptr cres = func(c);
	nnet::constant* cres_c = dynamic_cast<nnet::constant*>(cres.get());
	ASSERT_NE(nullptr, cres_c);
	EXPECT_TRUE(tensorshape_equal(shape, cres_c->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	std::vector<double> result = nnet::expose<double>(cres_c);
	assert(result.size() == constant_values.size()); // logical assertion
	primer(constant_values);
	for (size_t i = 0; i < constant_values.size(); i++)
	{
		double expectation = expect_forward(constant_values[i]);
		EXPECT_EQ(expectation, result[i]);
	}
}

static void binaryElemTest (FUZZ::fuzz_test* fuzzer, BINARY_VARS func,
	BINARY_VAR1 func1, BINARY_VAR2 func2, BINARY_SCALARS expect_forward,
	QUANARY_SCALARS expect_back)
{
	tensorshape shape = random_def_shape(fuzzer);
	size_t inn = shape.n_elems();
	rand_uniform rinit(2, 12);

	std::vector<size_t> shapelist = shape.as_list();
	size_t mutate_idx = fuzzer->get_int(1, "mutate_idx", {0, shapelist.size()-1})[0];
	shapelist[mutate_idx]++;
	tensorshape shape2 = shapelist;

	// matching pair
	std::vector<double> scalars = fuzzer->get_double(2, "scalars", {3, 50});
	variable var(shape, rinit, nnet::DOUBLE, "var");
	variable var2(shape, rinit, nnet::DOUBLE, "var2");

	// Behavior A000
	EXPECT_EQ(nullptr, func(varptr(nullptr), varptr(nullptr)));
	EXPECT_EQ(nullptr, func(varptr(&var), varptr(nullptr)));
	EXPECT_EQ(nullptr, func(varptr(nullptr), varptr(&var2)));

	variable var3(shape2, rinit, nnet::DOUBLE, "unmatching_in");
	varptr res = func(varptr(&var), varptr(&var2));
	varptr res1 = func1(varptr(&var), scalars[1]);
	varptr res2 = func2(scalars[0], varptr(&var2));

	// initialize
	var.initialize();
	var2.initialize();
	var3.initialize();
	std::vector<double> indata = expose<double>(&var);
	std::vector<double> indata2 = expose<double>(&var2);

	// compare data, shape must be equivalent, since we're testing elementary operations (A002)
	const tensor_double* tenn = dynamic_cast<const tensor_double*>(res->eval());
	const tensor_double* tenn1 = dynamic_cast<const tensor_double*>(res1->eval());
	const tensor_double* tenn2 = dynamic_cast<const tensor_double*>(res2->eval());
	std::vector<double> raw = tenn->expose();
	std::vector<double> raw1 = tenn1->expose();
	std::vector<double> raw2 = tenn2->expose();
	ASSERT_TRUE(tensorshape_equal(shape, tenn->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(shape, tenn1->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(shape, tenn2->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_EQ(raw.size(), inn);
	ASSERT_EQ(raw1.size(), inn);
	ASSERT_EQ(raw2.size(), inn);
	for (size_t i = 0; i < inn; i++)
	{
		double rawd = raw[i];
		double rawd1 = raw1[i];
		double rawd2 = raw2[i];

		double forwardd = expect_forward(indata[i], indata2[i]);
		double forwardd1 = expect_forward(indata[i], scalars[1]);
		double forwardd2 = expect_forward(scalars[0], indata2[i]);

		// allow some error in case of rounding error
		double errf = std::abs(forwardd - rawd);
		double errf1 = std::abs(forwardd1 - rawd1);
		double errf2 = std::abs(forwardd2 - rawd2);
		EXPECT_GE(epi, errf);
		EXPECT_GE(epi, errf1);
		EXPECT_GE(epi, errf2);
	}

	const tensor_double* backtens1 = dynamic_cast<const tensor_double*>(res->derive(&var)->eval());
	const tensor_double* backtens2 = dynamic_cast<const tensor_double*>(res->derive(&var2)->eval());
	const tensor_double* back1tens = dynamic_cast<const tensor_double*>(res1->derive(&var)->eval());
	const tensor_double* back2tens = dynamic_cast<const tensor_double*>(res2->derive(&var2)->eval());
	std::vector<double> raw3 = backtens1->expose();
	std::vector<double> raw4 = backtens2->expose();
	std::vector<double> raw5 = back1tens->expose();
	std::vector<double> raw6 = back2tens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, backtens1->get_shape()) || raw3.size() == 1) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(shape, backtens2->get_shape()) || raw4.size() == 1) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(shape, back1tens->get_shape()) || raw5.size() == 1) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(shape, back2tens->get_shape()) || raw6.size() == 1) <<
		sprintf("expecting shape %p, got %p", &shape, );
	if (raw3.size() == 1)
	{
		double rawd = raw3[0];
		double backd = expect_back(indata[0], indata2[0], 1, 0);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawd);
		EXPECT_GE(epi, errb);
	}
	if (raw4.size() == 1)
	{
		double rawd = raw4[0];
		double backd = expect_back(indata[0], indata2[0], 0, 1);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawd);
		EXPECT_GE(epi, errb);
	}
	if (raw5.size() == 1)
	{
		double rawd = raw5[0];
		double backd = expect_back(indata[0], scalars[1], 1, 0);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawd);
		EXPECT_GE(epi, errb);
	}
	if (raw6.size() == 1)
	{
		double rawd = raw6[0];
		double backd = expect_back(scalars[0], indata2[0], 0, 1);
		// allow some error in case of rounding error
		double errb = std::abs(backd - rawd);
		EXPECT_GE(epi, errb);
	}

	// Behavior A003
	varptr bad1 = func(varptr(&var), varptr(&var3));
	varptr bad2 = func(varptr(&var3), varptr(&var2));

	EXPECT_THROW(bad1->eval(), std::exception);
	EXPECT_THROW(bad2->eval(), std::exception);

	// Behavior A012
	variable vs(fuzzer->get_double(1, "vs_value")[0]);

	varptr same = func(varptr(&var), varptr(&vs));
	varptr same2 = func(varptr(&var2), varptr(&vs));
	varptr same3 = func(varptr(&var3), varptr(&vs));

	EXPECT_TRUE(tensorshape_equal(same->get_shape(), var.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(same2->get_shape(), var2.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(same3->get_shape(), var3.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
}


TEST_F(ELEMENTARY, Abs_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return +in; },
	[](double var) { return +var; },
	[](double, double gvar) { return +gvar; });
}


TEST_F(ELEMENTARY, Neg_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return -in; },
	[](double var) { return -var; },
	[](double, double gvar) { return -gvar; });
}


TEST_F(ELEMENTARY, Sin_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return sin(in); },
	[](double var) { return sin(var); },
	[](double var, double gvar) { return gvar * cos(var); });
}


TEST_F(ELEMENTARY, Cos_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return cos(in); },
	[](double var) { return cos(var); },
	[](double var, double gvar) { return -gvar * sin(var); });

}


TEST_F(ELEMENTARY, Tan_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return tan(in); },
	[](double var) { return tan(var); },
	[](double var, double gvar)
	{
		double s = cos(var);
		return gvar / std::pow(s, 2);
	});
}


TEST_F(ELEMENTARY, Csc_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return csc(in); },
	[](double var) { return 1/sin(var); },
	[](double var, double gvar)
	{
		return -gvar / (sin(var) * tan(var));
	});
}


TEST_F(ELEMENTARY, Sec_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return sec(in); },
	[](double var) { return 1/cos(var); },
	[](double var, double gvar) { return gvar * tan(var) / cos(var); });
}


TEST_F(ELEMENTARY, Cot_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return cot(in); },
	[](double var) { return cos(var) / sin(var); },
	[](double var, double gvar)
	{
		double c = 1/sin(var);
		return -gvar * c * c;
	});
}


TEST_F(ELEMENTARY, Exp_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return exp(in); },
	[](double var) { return exp(var); },
	[](double var, double gvar) { return gvar * exp(var); });
}


TEST_F(ELEMENTARY, Root_A000ToA003)
{
 	unaryElemTest(this,
 	[](varptr in) { return sqrt(in); },
 	[](double var) { return std::sqrt(var); },
	[](double var, double gvar) { return gvar / (2.0 * std::sqrt(var)); });
}


TEST_F(ELEMENTARY, Round_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return round(in); },
	[](double var) { return std::round(var); },
	[](double, double gvar) { return std::round(gvar); });
}


TEST_F(ELEMENTARY, Log_A000ToA003)
{
	unaryElemTest(this,
	[](varptr in) { return log(in); },
	[](double var) { return std::log(var); },
	[](double var, double) { return 1 / var; });
}


TEST_F(ELEMENTARY, Pow_A000ToA003)
{
	double scalar = get_double(1, "scalar", {-21, 13})[0];
 	unaryElemTest(this,
 	[scalar](varptr in) { return pow(in, scalar); },
 	[scalar](double var) { return std::pow(var, scalar); },
	[scalar](double var, double gvar) { return scalar * gvar * pow(var, scalar-1); });
}


TEST_F(ELEMENTARY, Clip_A000ToA003)
{
	std::vector<double> limits = get_double(2, "limits", {-100, 200});
	double min = limits[0]> limits[1] ? limits[1] : limits[0];
	double max = limits[0]> limits[1] ? limits[0] : limits[1];
	unaryElemTest(this,
	[max, min](varptr in) { return clip(in, min, max); },
	[max, min](double var)
	{
		if (var> max) var = max;
		else if (var < min) var = min;
		return var;
	},
	[max, min](double var, double gvar)
	{
		if (var> max) var = max;
		else if (var < min) var = min;
		return gvar * var;
	});
}


TEST_F(ELEMENTARY, ClipNorm_)
{
	// todo: add to behavior.txt
	double l2norm = 0;
	double cap = get_double(1, "cap", {0.1, 200})[0];
	unaryElemTest(this,
	[cap](varptr in) { return clip_norm(in, cap); },
	[cap, &l2norm](double var)
	{
		if (var <= l2norm)
		{
			var = var * cap / l2norm;
		}
		return var;
	},
	[cap, &l2norm](double var, double gvar)
	{
		if (var <= l2norm)
		{
			var = var * cap / l2norm;
		}
		return gvar * var;
	},
	[cap, &l2norm](std::vector<double>& vec)
	{
		l2norm = 0;
		for (double d : vec)
		{
			l2norm += d * d;
		}
		l2norm = std::sqrt(l2norm);
	});
}


TEST_F(ELEMENTARY, DISABLED_BinomSample_)
{
	// todo: implement + add to behavior.txt
}


TEST_F(ELEMENTARY, Condition_A000ToA003_A012)
{
	static std::vector<std::function<bool(double,double)>> conds = {
		[](double a, double b) { return a < b; },
		[](double a, double b) { return a> b; },
		[](double a, double b) { return a>= b; },
		[](double a, double b) { return a <= b; },
		[](double a, double b) { return a == b; },
		[](double a, double b) { return a != b; },
	};
	std::function<bool(double,double)> cond = conds[get_int(1, "condIdx", {0, conds.size() - 1})[0]];
	binaryElemTest(this,
	[cond](varptr a, varptr b)
	{
		return conditional(a, b, cond, "cond");
	},
	[cond](varptr a, double b)
	{
		return conditional(a, b, cond, "cond");
	},
	[cond](double a, varptr b)
	{
		return conditional(a, b, cond, "cond");
	},
	[cond](double a, double b) { return (double) cond(a, b); },
	[cond](double, double, double ga, double gb) { return (double) cond(ga, gb); });
}


TEST_F(ELEMENTARY, Eq_)
{
	// todo: add to behavior.txt
	std::function<bool(double,double)> cond = [](double a, double b) { return a == b; };
	binaryElemTest(this,
	[](varptr a, varptr b)
	{
		return eq(a, b);
	},
	[](varptr a, double b)
	{
		return eq(a, b);
	},
	[](double a, varptr b)
	{
		return eq(a, b);
	},
	[cond](double a, double b) { return (double) cond(a, b); },
	[cond](double, double, double ga, double gb) { return (double) cond(ga, gb); });
}


TEST_F(ELEMENTARY, Neq_)
{
	// todo: add to behavior.txt
	std::function<bool(double,double)> cond = [](double a, double b) { return a != b; };
	binaryElemTest(this,
	[](varptr a, varptr b)
	{
		return neq(a, b);
	},
	[](varptr a, double b)
	{
		return neq(a, b);
	},
	[](double a, varptr b)
	{
		return neq(a, b);
	},
	[cond](double a, double b) { return (double) cond(a, b); },
	[cond](double, double, double ga, double gb) { return (double) cond(ga, gb); });
}


TEST_F(ELEMENTARY, Add_A000ToA004_A012)
{
	binaryElemTest(this,
	[](varptr a, varptr b) { return a+b; },
	[](varptr a, double b) { return a+b; },
	[](double a, varptr b) { return a+b; },
	[](double a, double b) { return a+b; },
	[](double, double, double ga, double gb) { return ga+gb; });

	tensorshape shape = random_def_shape(this);
	rand_uniform rinit(2, 12);
	varptr zero = constant::get<double>(0.0);
	varptr one = constant::get<double>(1.0);
	variable var(shape, rinit, nnet::DOUBLE, "var");
	variable var2(shape, rinit, nnet::DOUBLE, "var2");

	// Behavior A004
	varptr samev1 = varptr(&var) +  0.0;
	varptr samev2 = 0.0 + varptr(&var2);
	varptr samev12 = varptr(&var) + varptr(zero);
	varptr samev22 = varptr(zero) + varptr(&var2);

	EXPECT_EQ(&var, samev1.get());
	EXPECT_EQ(&var2, samev2.get());
	EXPECT_EQ(&var, samev12.get());
	EXPECT_EQ(&var2, samev22.get());

	varptr wn = varptr(zero) + 1.0;
	varptr wn2 = 1.0 + varptr(zero);
	varptr to = varptr(one) + 1.0;
	varptr to2 = 1.0 + varptr(one);
	constant* wunres = dynamic_cast<constant*>(wn.get());
	constant* wunres2 = dynamic_cast<constant*>(wn2.get());
	constant* toores = dynamic_cast<constant*>(to.get());
	constant* toores2 = dynamic_cast<constant*>(to2.get());

	ASSERT_NE(nullptr, wunres);
	ASSERT_NE(nullptr, wunres2);
	ASSERT_NE(nullptr, toores);
	ASSERT_NE(nullptr, toores2);
	EXPECT_TRUE(*wunres == 1.0);
	EXPECT_TRUE(*wunres2 == 1.0);
	EXPECT_TRUE(*toores == 2.0);
	EXPECT_TRUE(*toores2 == 2.0);

	// never consumed by a node
	delete zero;
	delete one;
}


TEST_F(ELEMENTARY, Sub_A000ToA003_A012_A005)
{
	binaryElemTest(this,
	[](varptr a, varptr b) { return a-b; },
	[](varptr a, double b) { return a-b; },
	[](double a, varptr b) { return a-b; },
	[](double a, double b) { return a-b; },
	[](double, double, double ga, double gb) { return ga-gb; });

	tensorshape shape = random_def_shape(this);
	size_t inn = shape.n_elems();
	rand_uniform rinit(2, 12);
	varptr zero = constant::get<double>(0.0);
	varptr one = constant::get<double>(1.0);
	variable var(shape, rinit, nnet::DOUBLE, "var");
	variable var2(shape, rinit, nnet::DOUBLE, "var2");

	// Behavior A005
	varptr samev1 = varptr(&var) -  0.0;
	varptr samenv2 = 0.0 - varptr(&var2);
	varptr samev12 = varptr(&var) - varptr(zero);
	varptr samenv22 = varptr(zero) - varptr(&var2);

	EXPECT_EQ(&var, samev1.get());
	EXPECT_EQ(&var, samev12.get());

	// initialize
	var2.initialize();
	std::vector<double> indata2 = expose<double>(&var2);

	const tensor_double* rawtens = dynamic_cast<const tensor_double*>(samenv2->eval());
	const tensor_double* rawtens2 = dynamic_cast<const tensor_double*>(samenv22->eval());
	std::vector<double> rawf = rawtens->expose();
	std::vector<double> rawf2 = rawtens2->expose();
	ASSERT_TRUE(tensorshape_equal(shape, rawtens->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(shape, rawtens2->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_EQ(rawf.size(), inn);
	ASSERT_EQ(rawf2.size(), inn);
	for (size_t i = 0; i < inn; i++)
	{
		EXPECT_EQ(-indata2[i], rawf[i]);
		EXPECT_EQ(-indata2[i], rawf2[i]);
	}

	varptr ng = varptr(zero) - 1.0;
	varptr wn = 1.0 - varptr(zero);
	varptr zro = varptr(one) - 1.0;
	varptr zro2 = 1.0 - varptr(one);
	constant* negres = dynamic_cast<constant*>(ng.get());
	constant* wunres = dynamic_cast<constant*>(wn.get());
	constant* zerores = dynamic_cast<constant*>(zro.get());
	constant* zerores2 = dynamic_cast<constant*>(zro2.get());

	ASSERT_NE(nullptr, negres);
	ASSERT_NE(nullptr, wunres);
	ASSERT_NE(nullptr, zerores);
	ASSERT_NE(nullptr, zerores2);
	EXPECT_TRUE(*negres == -1.0);
	EXPECT_TRUE(*wunres == 1.0);
	EXPECT_TRUE(*zerores == 0.0);
	EXPECT_TRUE(*zerores2 == 0.0);

	// never consumed by a node
	delete zero;
	delete one;
}


TEST_F(ELEMENTARY, Mul_A000ToA003_A012_A006ToA007)
{
	binaryElemTest(this,
	[](varptr a, varptr b) { return a * b; },
	[](varptr a, double b) { return a * b; },
	[](double a, varptr b) { return a * b; },
	[](double a, double b) { return a * b; },
	[](double a, double b, double ga, double gb) { return ga*b+gb*a; });

	tensorshape shape = random_def_shape(this);
	rand_uniform rinit(2, 12);
	varptr zero = constant::get<double>(0.0);
	varptr one = constant::get<double>(1.0);
	varptr two = constant::get<double>(2.0);
	variable var(shape, rinit, nnet::DOUBLE, "var");
	variable var2(shape, rinit, nnet::DOUBLE, "var2");

	// Behavior A006
	varptr zaro = varptr(&var) *  0.0;
	varptr zaro2 = 0.0 * varptr(&var2);
	varptr zaro3 = varptr(&var) * varptr(zero);
	varptr zaro4 = varptr(zero) * varptr(&var2);
	std::vector<double> exp01 = expose<double>(zaro);
	std::vector<double> exp02 = expose<double>(zaro2);
	std::vector<double> exp03 = expose<double>(zaro3);
	std::vector<double> exp04 = expose<double>(zaro4);
	ASSERT_EQ((size_t) 1, exp01.size());
	ASSERT_EQ((size_t) 1, exp02.size());
	ASSERT_EQ((size_t) 1, exp03.size());
	ASSERT_EQ((size_t) 1, exp04.size());
	EXPECT_EQ(0, exp01.at(0));
	EXPECT_EQ(0, exp02.at(0));
	EXPECT_EQ(0, exp03.at(0));
	EXPECT_EQ(0, exp04.at(0));

	// Behavior A007
	varptr samev1 = varptr(&var) * 1.0;
	varptr samev2 = 1.0 * varptr(&var2);
	varptr samev12 = varptr(&var) * varptr(one);
	varptr samev22 = varptr(one) * varptr(&var2);

	EXPECT_EQ(&var, samev1.get());
	EXPECT_EQ(&var2, samev2.get());
	EXPECT_EQ(&var, samev12.get());
	EXPECT_EQ(&var2, samev22.get());

	varptr zro = varptr(zero) * 1.0;
	varptr zro2 = 1.0 * varptr(zero);
	varptr wn = varptr(one) * 1.0;
	varptr wn2 = 1.0 * varptr(one);
	varptr to = varptr(two) * 1.0;
	varptr to2 = 1.0 * varptr(two);
	constant* zerores = dynamic_cast<constant*>(zro.get());
	constant* zerores2 = dynamic_cast<constant*>(zro2.get());
	constant* wunres = dynamic_cast<constant*>(wn.get());
	constant* wunres2 = dynamic_cast<constant*>(wn2.get());
	constant* toores = dynamic_cast<constant*>(to.get());
	constant* toores2 = dynamic_cast<constant*>(to2.get());

	ASSERT_NE(nullptr, zerores);
	ASSERT_NE(nullptr, zerores2);
	ASSERT_NE(nullptr, wunres);
	ASSERT_NE(nullptr, wunres2);
	ASSERT_NE(nullptr, toores);
	ASSERT_NE(nullptr, toores2);
	EXPECT_TRUE(*zerores == 0.0);
	EXPECT_TRUE(*zerores2 == 0.0);
	EXPECT_TRUE(*wunres == 1.0);
	EXPECT_TRUE(*wunres2 == 1.0);
	EXPECT_TRUE(*toores == 2.0);
	EXPECT_TRUE(*toores2 == 2.0);

	// never consumed by a node
	delete zero;
	delete one;
	delete two;
}


TEST_F(ELEMENTARY, Div_A000ToA003_A012_A008ToA009)
{
	binaryElemTest(this,
	[](varptr a, varptr b) { return a/b; },
	[](varptr a, double b) { return a/b; },
	[](double a, varptr b) { return a/b; },
	[](double a, double b) { return a/b; },
	[](double a, double b, double ga, double gb) { return (ga*b-gb*a)/(b*b); });

	tensorshape shape = random_def_shape(this);
	rand_uniform rinit(2, 12);
	varptr zero = constant::get<double>(0.0);
	varptr one = constant::get<double>(1.0);
	varptr two = constant::get<double>(2.0);
	variable var(shape, rinit, nnet::DOUBLE, "var");
	variable var2(shape, rinit, nnet::DOUBLE, "var2");

	// Behavior A006
	varptr zaro = 0.0 / varptr(&var2);
	varptr zaro2 = varptr(zero) * varptr(&var2);
	EXPECT_THROW(varptr(&var) /  0.0, std::logic_error);
	EXPECT_THROW(1.0 / varptr(zero), std::logic_error);
	EXPECT_THROW(varptr(&var) / varptr(zero), std::logic_error);

	std::vector<double> exp01 = expose<double>(zaro);
	std::vector<double> exp02 = expose<double>(zaro2);
	ASSERT_EQ((size_t) 1, exp01.size());
	ASSERT_EQ((size_t) 1, exp02.size());
	EXPECT_EQ(0, exp01.at(0));
	EXPECT_EQ(0, exp02.at(0));

	// Behavior A007
	varptr samev1 = varptr(&var) / 1.0;
	varptr samev12 = varptr(&var) / varptr(one);

	EXPECT_EQ(&var, samev1.get());
	EXPECT_EQ(&var, samev12.get());

	varptr zro = varptr(zero) / 1.0;
	varptr wn = varptr(one) / 1.0;
	varptr wn2 = 1.0 / varptr(one);
	varptr hf = 1.0 / varptr(two);
	constant* zerores = dynamic_cast<constant*>(zro.get());
	constant* wunres = dynamic_cast<constant*>(wn.get());
	constant* wunres2 = dynamic_cast<constant*>(wn2.get());
	constant* halfres = dynamic_cast<constant*>(hf.get());

	ASSERT_NE(nullptr, zerores);
	ASSERT_NE(nullptr, wunres);
	ASSERT_NE(nullptr, wunres2);
	ASSERT_NE(nullptr, halfres);
	EXPECT_TRUE(*zerores == 0.0);
	EXPECT_TRUE(*wunres == 1.0);
	EXPECT_TRUE(*wunres2 == 1.0);
	EXPECT_TRUE(*halfres == 0.5);

	// never consumed by a node
	delete zero;
	delete one;
	delete two;
}



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
	EXPECT_TRUE(tensorshape_equal(expectoshape, outshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );
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
		EXPECT_TRUE(tensorshape_equal(gradoshape, outgshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );
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
			std::vector<size_t> incoord = inshape.coord_from_idx(i);
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
			std::vector<size_t> incoord = inshape.coord_from_idx(i);
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
			std::vector<size_t> incoord = inshape.coord_from_idx(i);
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
			std::vector<size_t> incoord = inshape.coord_from_idx(i);
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
	ASSERT_TRUE(tensorshape_equal(shape, rawtens->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_EQ(rawf.size(), inn);
	for (size_t i = 0; i < inn; i++)
	{
		std::vector<size_t> coord = shape.coord_from_idx(i);
		coord[flipdim] = shapelist[flipdim] - coord[flipdim] - 1;
		double out = rawf[i];
		double in = indata[shape.flat_idx(coord)];
		EXPECT_EQ(in, out);
	}

	// todo: (test) flip back prop
}



class MATMUL : public FUZZ::fuzz_test {};


using namespace nnet;


using TWODV = std::vector<std::vector<signed> >;


TWODV create2D (std::vector<signed> juanD, tensorshape mats, bool transpose = false)
{
	std::vector<size_t> dims = mats.as_list();
	size_t C = dims[0];
	size_t R = dims[1];
	TWODV res;

	size_t resC = transpose ? R : C;
	size_t resR = transpose ? C : R;
 	for (size_t y = 0; y < resR; y++)
	{
		res.push_back(std::vector<signed>(resC, 0));
	}

	for (size_t y = 0; y < R; y++)
	{
		for (size_t x = 0; x < C; x++)
		{
			size_t juan_coord = x + y * C;
			if (transpose)
			{
				res[x][y] = juanD[juan_coord];
			}
			else
			{
				res[y][x] = juanD[juan_coord];
			}
		}
	}
	return res;
}


bool freivald (FUZZ::fuzz_test* fuzzer, TWODV a, TWODV b, TWODV c)
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
		std::vector<signed> br;
		for (size_t y = 0, n = b.size(); y < n; y++)
		{
			signed bri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				bri += b[y][x] * r[x];
			}
			br.push_back(bri);
		}

		std::vector<signed> cr;
		for (size_t y = 0, n = c.size(); y < n; y++)
		{
			signed cri = 0;
			for (size_t x = 0; x < rlen; x++)
			{
				cri += c[y][x] * r[x];
			}
			cr.push_back(cri);
		}

		std::vector<signed> p;
		size_t n = a.size();
		for (size_t y = 0; y < n; y++)
		{
			signed ari = 0;
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
		if (!std::all_of(p.begin(), p.end(), [](signed d) { return d == 0; }))
			return false;
	}
	return true;
}


TEST_F(MATMUL, NullptrRet_C000)
{
	variable* zero = new variable(0);
	EXPECT_EQ(nullptr, matmul(nullptr, nullptr));
	EXPECT_EQ(nullptr, matmul(zero, nullptr));
	EXPECT_EQ(nullptr, matmul(nullptr, zero));
	delete zero;
}


TEST_F(MATMUL, Matmul_C001)
{
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = get_int(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform_int rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable A(shapeA, rinit, nnet::INT, "A"); // shape <m, n>
	variable B(shapeB, rinit, nnet::INT, "B"); // shape <k, m>
	variable tA(shapetA, rinit, nnet::INT, "tA");
	variable tB(shapetB, rinit, nnet::INT, "tB");

	// shapes of <k, n>
	varptr res = matmul(varptr(&A), varptr(&B));
	varptr restA = matmul(varptr(&tA), varptr(&B), true);
	varptr restB = matmul(varptr(&A), varptr(&tB), false, true);
	varptr resT = matmul(varptr(&tA), varptr(&tB), true, true);

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	tensorshape expectshape = std::vector<size_t>{dims[2], dims[1]};
	tensorshape resshape = res->get_shape();
	tensorshape restAshape = restA->get_shape();
	tensorshape restBshape = restB->get_shape();
	tensorshape resTshape = resT->get_shape();

	ASSERT_TRUE(tensorshape_equal(expectshape, resshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(expectshape, restAshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(expectshape, restBshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(expectshape, resTshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );

	TWODV matA = create2D(expose<signed>(&A), A.get_shape());
	TWODV matB = create2D(expose<signed>(&B), B.get_shape());
	TWODV mattA = create2D(expose<signed>(&tA), tA.get_shape(), true);
	TWODV mattB = create2D(expose<signed>(&tB), tB.get_shape(), true);

	TWODV matres = create2D(expose<signed>(res), resshape);
	TWODV matrestA = create2D(expose<signed>(restA), restAshape);
	TWODV matrestB = create2D(expose<signed>(restB), restBshape);
	TWODV matresT = create2D(expose<signed>(resT), resTshape);

	// Freivald's algorithm
	EXPECT_TRUE(freivald(this, matA, matB, matres));
	EXPECT_TRUE(freivald(this, mattA, matB, matrestA));
	EXPECT_TRUE(freivald(this, matA, mattB, matrestB));
	EXPECT_TRUE(freivald(this, mattA, mattB, matresT));

	// we delete top nodes, because this case is not testing for observer self-destruction
	delete res;
	delete restA;
	delete restB;
	delete resT;
}


// tests matrix multiplication but for n dimensions, matrix sizes reduced to 2-5, (we get at most 5x25 matmuls)
// todo: test
TEST_F(MATMUL, DISABLED_NDim_Matmul_C001)
{
}


TEST_F(MATMUL, Incompatible_C002)
{
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = get_int(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]+1};

	variable A(shapeA, rinit, nnet::DOUBLE, "A"); // shape <m, n>
	variable B(shapeB, rinit, nnet::DOUBLE, "B"); // shape <k, m+1>

	A.initialize();
	B.initialize();

	varptr bad = matmul(varptr(&A), varptr(&B));
	EXPECT_THROW(bad->eval(), std::logic_error);
}


TEST_F(MATMUL, Jacobian_C003)
{
	// we get at most 49 elements per matrix
	std::vector<size_t> dims = get_int(3, "dimensions<m,n,k>", {3, 7});
	rand_uniform rinit(0, 1);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable A(shapeA, rinit, nnet::DOUBLE, "A"); // shape <m, n>
	variable B(shapeB, rinit, nnet::DOUBLE, "B"); // shape <k, m>
	variable tA(shapetA, rinit, nnet::DOUBLE, "tA");
	variable tB(shapetB, rinit, nnet::DOUBLE, "tB");

	// shapes of <k, n>
	varptr res = sigmoid(varptr(matmul(varptr(&A), varptr(&B))));
	varptr restA = sigmoid(varptr(matmul(varptr(&tA), varptr(&B), true)));
	varptr restB = sigmoid(varptr(matmul(varptr(&A), varptr(&tB), false, true)));
	varptr resT = sigmoid(varptr(matmul(varptr(&tA), varptr(&tB), true, true)));

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	inode* dresA = res->derive(&A);
	inode* dresB = res->derive(&B);

	inode* drestAA = restA->derive(&tA);
	inode* drestAB = restA->derive(&B);

	inode* drestBA = restB->derive(&A);
	inode* drestBB = restB->derive(&tB);

	inode* dresTA = resT->derive(&tA);
	inode* dresTB = resT->derive(&tB);

	// requires on all elementary operations to be valid (not a great validation method...)
	// res = 1/(1+e^-(A@B))
	// dres = jacobian(sigmoid'(1))
	// where jacobian = {
	// 		sigmoid'(1) @ B^T for dA
	//		A^T @ sigmoid'(1) for dB
	// }
	// sigmoid' = sigmoid * (1 - sigmoid)
	varptr dsig_res = res * (1.0 - res);
	inode* fake_dresA = matmul(dsig_res, &B, false, true);
	inode* fake_dresB = matmul(&A, dsig_res, true);

	varptr dsig_restA = restA * (1.0 - restA);
	inode* fake_drestAA = transpose(matmul(dsig_restA, &B, false, true));
	inode* fake_drestAB = matmul(&tA, dsig_restA);

	varptr dsig_restB = restB * (1.0 - restB);
	inode* fake_drestBA = matmul(dsig_restB, &tB);
	inode* fake_drestBB = transpose(matmul(&A, dsig_restB, true, false));

	varptr dsig_resT = resT * (1.0 - resT);
	inode* fake_dresTA = transpose(matmul(dsig_resT, &tB));
	inode* fake_dresTB = transpose(matmul(&tA, dsig_resT));

	EXPECT_TRUE(tensorshape_equal(dresA->get_shape(), A.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(dresB->get_shape(), B.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(drestAA->get_shape(), tA.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(drestAB->get_shape(), B.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(drestBA->get_shape(), A.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(drestBB->get_shape(), tB.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(dresTA->get_shape(), tA.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(dresTB->get_shape(), tB.get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );

	EXPECT_TRUE(tensorshape_equal(dresA->get_shape(), fake_dresA->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(dresB->get_shape(), fake_dresB->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(drestAA->get_shape(), fake_drestAA->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(drestAB->get_shape(), fake_drestAB->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(drestBA->get_shape(), fake_drestBA->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(drestBB->get_shape(), fake_drestBB->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(dresTA->get_shape(), fake_dresTA->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );
	EXPECT_TRUE(tensorshape_equal(dresTB->get_shape(), fake_dresTB->get_shape())) <<
		sprintf("expecting shape %p, got %p", &shape, );

	std::vector<double> dresA_data = expose<double>(dresA);
	std::vector<double> dresB_data = expose<double>(dresB);
	std::vector<double> drestAA_data = expose<double>(drestAA);
	std::vector<double> drestAB_data = expose<double>(drestAB);
	std::vector<double> drestBA_data = expose<double>(drestBA);
	std::vector<double> drestBB_data = expose<double>(drestBB);
	std::vector<double> dresTA_data = expose<double>(dresTA);
	std::vector<double> dresTB_data = expose<double>(dresTB);

	std::vector<double> fake_dresA_data = expose<double>(fake_dresA);
	std::vector<double> fake_dresB_data = expose<double>(fake_dresB);
	std::vector<double> fake_drestAA_data = expose<double>(fake_drestAA);
	std::vector<double> fake_drestAB_data = expose<double>(fake_drestAB);
	std::vector<double> fake_drestBA_data = expose<double>(fake_drestBA);
	std::vector<double> fake_drestBB_data = expose<double>(fake_drestBB);
	std::vector<double> fake_dresTA_data = expose<double>(fake_dresTA);
	std::vector<double> fake_dresTB_data = expose<double>(fake_dresTB);

	// all a shapes should have the same number of elements
	double err_thresh = 0.0000001;
	for (size_t i = 0, n = dresA_data.size(); i < n; i++)
	{
		double dresAerr = std::abs(dresA_data[i] - fake_dresA_data[i]);
		double drestAAerr = std::abs(drestAA_data[i] - fake_drestAA_data[i]);
		double drestBAerr = std::abs(drestBA_data[i] - fake_drestBA_data[i]);
		double dresTAerr = std::abs(dresTA_data[i] - fake_dresTA_data[i]);
		EXPECT_GT(err_thresh, dresAerr);
		EXPECT_GT(err_thresh, drestAAerr);
		EXPECT_GT(err_thresh, drestBAerr);
		EXPECT_GT(err_thresh, dresTAerr);
	}
	for (size_t i = 0, n = dresB_data.size(); i < n; i++)
	{
		double dresBerr = std::abs(dresB_data[i] - fake_dresB_data[i]);
		double drestABerr = std::abs(drestAB_data[i] - fake_drestAB_data[i]);
		double drestBBerr = std::abs(drestBB_data[i] - fake_drestBB_data[i]);
		double dresTBerr = std::abs(dresTB_data[i] - fake_dresTB_data[i]);
		EXPECT_GT(err_thresh, dresBerr);
		EXPECT_GT(err_thresh, drestABerr);
		EXPECT_GT(err_thresh, drestBBerr);
		EXPECT_GT(err_thresh, dresTBerr);
	}
}


// tests large matrices sizes (100-112), 2D only
TEST_F(MATMUL, Strassen_C004)
{
	// we get at most 12996 elements per matrix
	std::vector<size_t> dims = get_int(3, "dimensions<m,n,k>", {STRASSEN_THRESHOLD, STRASSEN_THRESHOLD+12});
	rand_uniform_int rinit(-12, 12);

	tensorshape shapeA = std::vector<size_t>{dims[0], dims[1]};
	tensorshape shapeB = std::vector<size_t>{dims[2], dims[0]};
	tensorshape shapetA = std::vector<size_t>{dims[1], dims[0]}; // transpose A
	tensorshape shapetB = std::vector<size_t>{dims[0], dims[2]}; // transpose B

	variable A(shapeA, rinit, nnet::INT, "A"); // shape <m, n>
	variable B(shapeB, rinit, nnet::INT, "B"); // shape <k, m>
	variable tA(shapetA, rinit, nnet::INT, "tA");
	variable tB(shapetB, rinit, nnet::INT, "tB");

	A.initialize();
	B.initialize();
	tA.initialize();
	tB.initialize();

	// shapes of <k, n>
//	clock_t t = clock();
	varptr res = matmul(varptr(&A), varptr(&B));
//	const double work_time1 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr restA = matmul(varptr(&tA), varptr(&B), true);
//	const double work_time2 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr restB = matmul(varptr(&A), varptr(&tB), false, true);
//	const double work_time3 = (clock() - t) / double(CLOCKS_PER_SEC);

//	t = clock();
	varptr resT = matmul(varptr(&tA), varptr(&tB), true, true);
//	const double work_time4 = (clock() - t) / double(CLOCKS_PER_SEC);
//	ASSERT_GT(0.3, work_time1);
//	ASSERT_GT(0.3, work_time2);
//	ASSERT_GT(0.3, work_time3);
//	ASSERT_GT(0.3, work_time4);

	tensorshape expectshape = std::vector<size_t>{dims[2], dims[1]};
	tensorshape resshape = res->get_shape();
	tensorshape restAshape = restA->get_shape();
	tensorshape restBshape = restB->get_shape();
	tensorshape resTshape = resT->get_shape();

	ASSERT_TRUE(tensorshape_equal(expectshape, resshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(expectshape, restAshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(expectshape, restBshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );
	ASSERT_TRUE(tensorshape_equal(expectshape, resTshape)) <<
		sprintf("expecting shape %p, got %p", &shape, );

	TWODV matA = create2D(expose<signed>(&A), A.get_shape());
	TWODV matB = create2D(expose<signed>(&B), B.get_shape());
	TWODV mattA = create2D(expose<signed>(&tA), tA.get_shape(), true);
	TWODV mattB = create2D(expose<signed>(&tB), tB.get_shape(), true);

	TWODV matres = create2D(expose<signed>(res), resshape);
	TWODV matrestA = create2D(expose<signed>(restA), restAshape);
	TWODV matrestB = create2D(expose<signed>(restB), restBshape);
	TWODV matresT = create2D(expose<signed>(resT), resTshape);
	// Freivald's algorithm

	EXPECT_TRUE(freivald(this, matA, matB, matres));
	EXPECT_TRUE(freivald(this, mattA, matB, matrestA));
	EXPECT_TRUE(freivald(this, matA, mattB, matrestB));
	EXPECT_TRUE(freivald(this, mattA, mattB, matresT));

	// we delete top nodes, because this case is not testing for observer self-destruction
	delete res;
	delete restA;
	delete restB;
	delete resT;
}


#endif /* DISABLE_BIND_TEST */


#endif /* DISABLE_OPERATE_MODULE_TESTS */
