//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_OPERATION_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "include/graph/leaf/variable.hpp"
#include "include/graph/operations/operations.hpp"

#include "tests/include/util_test.h"
#include "tests/include/fuzz.h"


#ifndef DISABLE_ELEMENTARY_TEST


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
	UNARY_SCALAR expect_forward, BINARY_SCALARS expect_back)
{
	tensorshape shape = random_def_shape(fuzzer);
	size_t inn = shape.n_elems();
	rand_uniform<double> rinit(2, 12);

	variable var(shape, rinit, "unar_var");
	varptr res = func(varptr(&var));

	// Behavior A000
	EXPECT_EQ(nullptr, func(varptr(nullptr)));

	// initialize
	var.initialize();
	std::vector<double> indata = expose<double>(&var);

	// compare data, shape must be equivalent, since we're testing elementary operations (Behavior A002)
	const tensor<double>* rawtens = res->eval();
	std::vector<double> rawf = rawtens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, rawtens->get_shape()));
	ASSERT_EQ(rawf.size(), inn);
	for (size_t i = 0; i < inn; i++)
	{
		double rawd = rawf[i];
		double forwardd = expect_forward(indata[i]);
		// allow some error in case of rounding error
		double errf = std::abs(forwardd - rawd);
		EXPECT_GE(epi, errf);
	}

	const tensor<double>* backtens = res->derive(&var)->eval();
	std::vector<double> rawb = backtens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, backtens->get_shape()) || rawb.size() == 1);
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
	nnet::constant* c = nnet::constant::get(constant_values, shape);
	nnet::varptr cres = func(c);
	nnet::constant* cres_c = dynamic_cast<nnet::constant*>(cres.get());
	ASSERT_NE(nullptr, cres_c);
	EXPECT_TRUE(tensorshape_equal(shape, cres_c->get_shape()));
	std::vector<double> result = nnet::expose<double>(cres_c);
	assert(result.size() == constant_values.size()); // logical assertion
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
	rand_uniform<double> rinit(2, 12);

	std::vector<size_t> shapelist = shape.as_list();
	size_t mutate_idx = fuzzer->get_int(1, "mutate_idx", {0, shapelist.size()-1})[0];
	shapelist[mutate_idx]++;
	tensorshape shape2 = shapelist;

	// matching pair
	std::vector<double> scalars = fuzzer->get_double(2, "scalars", {3, 50});
	variable var(shape, rinit, "var");
	variable var2(shape, rinit, "var2");

	// Behavior A000
	EXPECT_EQ(nullptr, func(varptr(nullptr), varptr(nullptr)));
	EXPECT_EQ(nullptr, func(varptr(&var), varptr(nullptr)));
	EXPECT_EQ(nullptr, func(varptr(nullptr), varptr(&var2)));

	variable var3(shape2, rinit, "unmatching_in");
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
	const tensor<double>* tenn = res->eval();
	const tensor<double>* tenn1 = res1->eval();
	const tensor<double>* tenn2 = res2->eval();
	std::vector<double> raw = tenn->expose();
	std::vector<double> raw1 = tenn1->expose();
	std::vector<double> raw2 = tenn2->expose();
	ASSERT_TRUE(tensorshape_equal(shape, tenn->get_shape()));
	ASSERT_TRUE(tensorshape_equal(shape, tenn1->get_shape()));
	ASSERT_TRUE(tensorshape_equal(shape, tenn2->get_shape()));
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

	const tensor<double>* backtens1 = res->derive(&var)->eval();
	const tensor<double>* backtens2 = res->derive(&var2)->eval();
	const tensor<double>* back1tens = res1->derive(&var)->eval();
	const tensor<double>* back2tens = res2->derive(&var2)->eval();
	std::vector<double> raw3 = backtens1->expose();
	std::vector<double> raw4 = backtens2->expose();
	std::vector<double> raw5 = back1tens->expose();
	std::vector<double> raw6 = back2tens->expose();
	ASSERT_TRUE(tensorshape_equal(shape, backtens1->get_shape()) || raw3.size() == 1);
	ASSERT_TRUE(tensorshape_equal(shape, backtens2->get_shape()) || raw4.size() == 1);
	ASSERT_TRUE(tensorshape_equal(shape, back1tens->get_shape()) || raw5.size() == 1);
	ASSERT_TRUE(tensorshape_equal(shape, back2tens->get_shape()) || raw6.size() == 1);
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

	EXPECT_TRUE(tensorshape_equal(same->get_shape(), var.get_shape()));
	EXPECT_TRUE(tensorshape_equal(same2->get_shape(), var2.get_shape()));
	EXPECT_TRUE(tensorshape_equal(same3->get_shape(), var3.get_shape()));
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


TEST_F(ELEMENTARY, ClipVal_A000ToA003)
{
	std::vector<double> limits = get_double(2, "limits", {-100, 200});
	double min = limits[0]> limits[1] ? limits[1] : limits[0];
	double max = limits[0]> limits[1] ? limits[0] : limits[1];
	unaryElemTest(this,
	[max, min](varptr in) { return clip_val(in, min, max); },
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
		return conditional(a, b, cond, "lessthan");
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
	rand_uniform<double> rinit(2, 12);
	varptr zero = constant::get(0.0);
	varptr one = constant::get(1.0);
	variable var(shape, rinit, "var");
	variable var2(shape, rinit, "var2");

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
	rand_uniform<double> rinit(2, 12);
	varptr zero = constant::get(0.0);
	varptr one = constant::get(1.0);
	variable var(shape, rinit, "var");
	variable var2(shape, rinit, "var2");

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

	const tensor<double>* rawtens = samenv2->eval();
	const tensor<double>* rawtens2 = samenv22->eval();
	std::vector<double> rawf = rawtens->expose();
	std::vector<double> rawf2 = rawtens2->expose();
	ASSERT_TRUE(tensorshape_equal(shape, rawtens->get_shape()));
	ASSERT_TRUE(tensorshape_equal(shape, rawtens2->get_shape()));
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
	rand_uniform<double> rinit(2, 12);
	varptr zero = constant::get(0.0);
	varptr one = constant::get(1.0);
	varptr two = constant::get(2.0);
	variable var(shape, rinit, "var");
	variable var2(shape, rinit, "var2");

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
	rand_uniform<double> rinit(2, 12);
	varptr zero = constant::get(0.0);
	varptr one = constant::get(1.0);
	varptr two = constant::get(2.0);
	variable var(shape, rinit, "var");
	variable var2(shape, rinit, "var2");

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


#endif /* DISABLE_ELEMENTARY_TEST */


#endif /* DISABLE_OPERATION_MODULE_TESTS */
