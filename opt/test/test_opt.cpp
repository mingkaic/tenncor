
#ifndef DISABLE_OPTIMIZATION_TEST


#include "gtest/gtest.h"

#include "dbg/stream/ade_csv.hpp"

#include "testutil/tutil.hpp"

#include "exam/exam.hpp"

#include "ead/generated/api.hpp"
#include "ead/parse.hpp"
#include "ead/constant.hpp"

#include "opt/optimize.hpp"


TEST(OPTIMIZE, CalcConstants)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"special_var"));

	ead::NodeptrT<double> two =
		ead::make_constant_scalar<double>(2, ade::Shape());
	ead::NodeptrT<double> three =
		ead::make_constant_scalar<double>(3, ade::Shape());
	ead::NodeptrT<double> four =
		ead::make_constant_scalar<double>(4, ade::Shape());

	opt::OptCtx empty_rules = ead::parse<double>("");

	{
		auto vfunc = tenncor::sin(var);
		auto opted = opt::optimize({vfunc->get_tensor()}, empty_rules);
		ASSERT_EQ(1, opted.size());
		// expect optimized vfunc to remain the same
		EXPECT_GRAPHEQ(
			"(SIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	{
		auto cfunc = tenncor::sin(two);
		auto opted = opt::optimize({cfunc->get_tensor()}, empty_rules);
		ASSERT_EQ(1, opted.size());
		// expect optimized cfunc to be sin(2)
		EXPECT_GRAPHEQ(
			(fmts::sprintf("(constant:%f[1\\1\\1\\1\\1\\1\\1\\1])", std::sin(2))),
			opted[0]);
	}

	{
		auto adv_func = tenncor::mul(
			tenncor::add(tenncor::sin(var), tenncor::sin(two)),
			tenncor::pow(three, four));
		auto opted = opt::optimize({adv_func->get_tensor()}, empty_rules);
		ASSERT_EQ(1, opted.size());
		// expect optimized adv_func to be (sin(var) + sin(2)) * 81
		EXPECT_GRAPHEQ(
			(fmts::sprintf(
				"(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" `--(ADD[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" |   `--(SIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" |   |   `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" |   `--(constant:%g[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" `--(constant:%g[1\\1\\1\\1\\1\\1\\1\\1])",
				std::sin(2), std::pow(3, 4))),
			opted[0]);

		// since the root never changed, expect change to be inline
		EXPECT_GRAPHEQ(
			(fmts::sprintf(
				"(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" `--(ADD[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" |   `--(SIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" |   |   `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" |   `--(constant:%g[1\\1\\1\\1\\1\\1\\1\\1])\n"
				" `--(constant:%g[1\\1\\1\\1\\1\\1\\1\\1])",
				std::sin(2), std::pow(3, 4))),
			adv_func->get_tensor());
	}
}


TEST(OPTIMIZE, PruneZeroSingles)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"special_var"));

	ead::NodeptrT<double> zero =
		ead::make_constant_scalar<double>(0, ade::Shape());

	opt::OptCtx rules = ead::parse_file<double>("cfg/optimizations.rules");

	{
		auto wunfunc = tenncor::pow(var, zero);
		auto zrofunc = tenncor::pow(zero, var);
		auto opted = opt::optimize({
			wunfunc->get_tensor(),
			zrofunc->get_tensor(),
		}, rules);
		ASSERT_EQ(2, opted.size());
		// expect both optimized wunfunc to be 1
		EXPECT_GRAPHEQ("(constant:1[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);

		// expect both optimized zrofunc to be 0
		EXPECT_GRAPHEQ("(constant:0[1\\1\\1\\1\\1\\1\\1\\1])", opted[1]);
	}

	{
		auto lvfunc = tenncor::add(var, zero);
		auto rvfunc = tenncor::add(zero, var);
		auto opted = opt::optimize({
			lvfunc->get_tensor(),
			rvfunc->get_tensor(),
		}, rules);
		ASSERT_EQ(2, opted.size());
		// expect both optimized l and r vfuncs to be var
		std::string expect = "(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto lzero = tenncor::mul(var, zero);
		auto rzero = tenncor::mul(zero, var);
		auto opted = opt::optimize({
			lzero->get_tensor(),
			rzero->get_tensor(),
		}, rules);
		ASSERT_EQ(2, opted.size());
		// expect both optimized l and r zeros to be 0
		std::string expect = "(constant:0[1\\1\\1\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto posvar = tenncor::sub(var, zero);
		auto negvar = tenncor::sub(zero, var);
		auto opted = opt::optimize({
			posvar->get_tensor(),
			negvar->get_tensor(),
		}, rules);
		ASSERT_EQ(2, opted.size());
		// expect optimized posvar to be var
		EXPECT_GRAPHEQ("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
			opted[0]);

		// expect optimized negvar to be -var
		EXPECT_GRAPHEQ(
			"(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])", opted[1]);
	}

	{
		auto divz = tenncor::div(zero, var);
		auto opted = opt::optimize({divz->get_tensor()}, rules);
		ASSERT_EQ(1, opted.size());
		// expect optimized divz to be zero
		EXPECT_GRAPHEQ("(constant:0[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
	}

	{
		auto no_opt = tenncor::max(zero, var);
		auto opted = opt::optimize({no_opt->get_tensor()}, rules);
		ASSERT_EQ(1, opted.size());
		// expect optimized not_opt to remain the same
		EXPECT_GRAPHEQ(
			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(constant:0[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
	}
}


TEST(OPTIMIZE, PruneZeroGraph)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(), "var"));
	ead::NodeptrT<double> var2 = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(), "var2"));

	ead::NodeptrT<double> zero =
		ead::make_constant_scalar<double>(0, ade::Shape());

	opt::OptCtx rules = ead::parse_file<double>("cfg/optimizations.rules");

	auto got1 = tenncor::cos(zero);
	auto got3 = tenncor::add(zero, var2);
	auto gotn1 = tenncor::sub(zero, var);
	auto got2 = tenncor::sub(var2, zero);
	auto got22 = tenncor::max(var2, zero);

	auto too = tenncor::add(zero, tenncor::mul(got1, got22));
	auto got11 = tenncor::pow(got2, zero);

	auto m = tenncor::min(tenncor::max(got22, got1), tenncor::lt(too, got11));
	auto nocascades = tenncor::sub(tenncor::pow(m, tenncor::div(got3, gotn1)), got2);

	auto opted = opt::optimize({nocascades->get_tensor()}, rules);
	ASSERT_EQ(1, opted.size());
	EXPECT_GRAPHEQ("(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |   |   `--(variable:var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |   |   `--(constant:0[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   `--(LT[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |   `--(variable:var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |   `--(constant:0[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(variable:var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |           `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:var2[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);

	auto got0 = tenncor::tan(zero);
	opted = opt::optimize({tenncor::pow(nocascades, got0)->get_tensor()}, rules);
	ASSERT_EQ(1, opted.size());
	EXPECT_GRAPHEQ("(constant:1[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
}


TEST(OPTIMIZE, PruneOneSingles)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"special_var"));

	ead::NodeptrT<double> one =
		ead::make_constant_scalar<double>(1, ade::Shape());

	opt::OptCtx rules = ead::parse_file<double>("cfg/optimizations.rules");

	{
		auto vfunc = tenncor::pow(var, one);
		auto wunfunc = tenncor::pow(one, var);
		auto opted = opt::optimize({
			vfunc->get_tensor(),
			wunfunc->get_tensor(),
		}, rules);
		ASSERT_EQ(2, opted.size());
		// expect optimized vfunc to be 1
		EXPECT_GRAPHEQ("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
			opted[0]);

		// expect optimized wunfunc to be 1
		EXPECT_GRAPHEQ("(constant:1[1\\1\\1\\1\\1\\1\\1\\1])", opted[1]);
	}

	{
		auto lvfunc = tenncor::mul(var, one);
		auto rvfunc = tenncor::mul(one, var);
		auto opted = opt::optimize({
			lvfunc->get_tensor(),
			rvfunc->get_tensor(),
		}, rules);
		ASSERT_EQ(2, opted.size());
		// expect both optimized l and r vfuncs to be var
		std::string expect = "(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto nomer = tenncor::div(var, one);
		auto opted = opt::optimize({nomer->get_tensor()}, rules);
		ASSERT_EQ(1, opted.size());
		// expect optimized nomer to be var
		EXPECT_GRAPHEQ("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	{
		auto wun = tenncor::div(var, var);
		auto opted = opt::optimize({wun->get_tensor()}, rules);
		ASSERT_EQ(1, opted.size());
		// expect optimized wun to be 1
		EXPECT_GRAPHEQ("(constant:1[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
	}

	{
		auto no_opt = tenncor::max(one, var);
		auto opted = opt::optimize({no_opt->get_tensor()}, rules);
		// expect optimized no_opt to remain the same
		EXPECT_GRAPHEQ(
			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
	}
}


TEST(OPTIMIZE, PruneOneGraph)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"var"));

	ead::NodeptrT<double> one =
		ead::make_constant_scalar<double>(1, ade::Shape());

	opt::OptCtx rules = ead::parse_file<double>("cfg/optimizations.rules");

	auto got0 = tenncor::log(one);
	auto got1 = tenncor::sqrt(one);
	auto got3 = tenncor::mul(one, var);
	auto got00 = tenncor::pow(one, var);
	auto got = tenncor::max(var, one);

	auto too = tenncor::add(got1, tenncor::mul(got0, got00));
	auto got11 = tenncor::pow(var, one);

	auto m = tenncor::min(tenncor::max(got1, too), got11);
	auto root = tenncor::sub(tenncor::pow(m, tenncor::div(got3, got)), var);

	auto opted = opt::optimize({root->get_tensor()}, rules);
	ASSERT_EQ(1, opted.size());
	EXPECT_GRAPHEQ(
		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |           `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |           `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
}


TEST(OPTIMIZE, PruneOpSingles)
{
	ead::NodeptrT<double> zero = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(), "special_var0"));
	ead::NodeptrT<double> one = ead::convert_to_node(
		ead::make_variable_scalar<double>(1, ade::Shape({2, 3}), "special_var"));

	opt::OptCtx rules = ead::parse_file<double>("cfg/optimizations.rules");

	// merge redundent double reduced argument for empty shape
	{
		auto opted = opt::optimize({
			tenncor::reduce_sum(tenncor::reduce_sum(zero))->get_tensor(),
		}, rules);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ("(variable:special_var0[1\\1\\1\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	// merge redundent double reduced argument for non-empty shape
	{
		auto opted = opt::optimize({
			tenncor::reduce_sum(tenncor::reduce_sum(one))->get_tensor(),
		}, rules);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ(
			"(REDUCE_SUM[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(variable:special_var[2\\3\\1\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	// don't merge non-redundent double reduced argument
	{
		auto opted = opt::optimize({
			tenncor::reduce_sum(tenncor::reduce_sum(one, 1), 0)->get_tensor(),
		}, rules);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ(
			"(REDUCE_SUM[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(REDUCE_SUM[2\\1\\1\\1\\1\\1\\1\\1])\n"
			"     `--(variable:special_var[2\\3\\1\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	// don't merge mul-reduced_add
	{
		auto opted = opt::optimize({
			tenncor::mul(tenncor::reduce_sum(one), zero)->get_tensor(),
		}, rules);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ(
			"(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(REDUCE_SUM[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" |   `--(variable:special_var[2\\3\\1\\1\\1\\1\\1\\1])\n"
			" `--(variable:special_var0[1\\1\\1\\1\\1\\1\\1\\1])",
			opted[0]);
	}
}


TEST(OPTIMIZE, PruneOpGraph)
{
	ead::NodeptrT<double> zero = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape({3, 4}), "special_var0"));
	ead::NodeptrT<double> one = ead::convert_to_node(
		ead::make_variable_scalar<double>(1, ade::Shape(), "special_var"));
	ead::NodeptrT<double> two = ead::convert_to_node(
		ead::make_variable_scalar<double>(2, ade::Shape(), "special_var2"));
	ead::NodeptrT<double> three = ead::convert_to_node(
		ead::make_variable_scalar<double>(3, ade::Shape(), "special_var3"));

	auto got1 = tenncor::cos(three);
	auto got3 = tenncor::mul(tenncor::mul(one, three), two);
	auto gotn1 = tenncor::sub(three, one);
	auto got2 = tenncor::sub(two, three);
	auto got22 = tenncor::min(two, three);

	auto too = tenncor::mul(tenncor::reduce_prod(tenncor::reduce_prod(zero, 0), 1),
		tenncor::reduce_prod(tenncor::mul(got1, got22)));
	auto got11 = tenncor::pow(got2, three);

	auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));

	opt::OptCtx rules = ead::parse_file<double>("cfg/optimizations.rules");

	auto opted = opt::optimize({
		tenncor::sub(tenncor::min(m, tenncor::div(got3, gotn1)), got2)->get_tensor(),
	}, rules);
	ASSERT_EQ(1, opted.size());
	auto root = opted[0];

	EXPECT_GRAPHEQ(
		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |   |   `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |   |   `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   |       `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |   `--(REDUCE_PROD[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |   |   `--(variable:special_var0[3\\4\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |   `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |       `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |       |   `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |       `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |           `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       |           `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |       `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |           `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |           |   `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |           |   `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   |           `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       |   `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       |   |   `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       |   |   `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       |   `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |       `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |           `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |           `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"     `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])",
		opted[0]);
}


TEST(OPTIMIZE, GroupSingles)
{
	ead::NodeptrT<double> one = ead::convert_to_node(
		ead::make_variable_scalar<double>(1, ade::Shape(), "special_var"));
	ead::NodeptrT<double> two = ead::convert_to_node(
		ead::make_variable_scalar<double>(2, ade::Shape(), "special_var2"));

	opt::OptCtx rules = ead::parse_file<double>("cfg/optimizations.rules");

	// mul and div and next to each level
	{
		auto opted = opt::optimize({
			tenncor::mul(tenncor::div(one, two), two)->get_tensor(),
		}, rules);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	// mul and div are separated by a level
	{
		auto opted = opt::optimize({
			tenncor::mul(tenncor::mul(tenncor::div(one, two), one), two)->get_tensor(),
		}, rules);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ(
			"(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
			opted[0]);
	}
}


TEST(OPTIMIZE, ReuseOpGraph)
{
	ead::NodeptrT<double> zero = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape()));
	ead::NodeptrT<double> one = ead::convert_to_node(
		ead::make_variable_scalar<double>(1, ade::Shape()));
	ead::NodeptrT<double> two = ead::convert_to_node(
		ead::make_variable_scalar<double>(2, ade::Shape()));

	ead::NodeptrT<double> root;
	{
		auto got1 = tenncor::cos(zero);
		auto got3 = tenncor::add(tenncor::add(one, zero), two);
		auto gotn1 = tenncor::sub(zero, one);
		auto got2 = tenncor::sub(two, zero);
		auto got22 = tenncor::max(two, zero);

		auto too = tenncor::add(zero, tenncor::mul(got1, got22));
		auto got11 = tenncor::pow(got2, zero);

		auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));
		root = tenncor::sub(tenncor::pow(m, tenncor::div(got3, gotn1)), got2);
	}

	ead::NodeptrT<double> subroot;
	{
		auto other_got1 = tenncor::cos(zero);
		auto got22 = tenncor::max(two, zero);
		subroot = tenncor::mul(other_got1, got22);
	}

	ead::NodeptrT<double> copyroot;
	{
		auto got1 = tenncor::cos(zero);
		auto got3 = tenncor::add(tenncor::add(one, zero), two);
		auto gotn1 = tenncor::sub(zero, one);
		auto got2 = tenncor::sub(two, zero);
		auto got22 = tenncor::max(two, zero);

		auto too = tenncor::add(zero, tenncor::mul(got1, got22));
		auto got11 = tenncor::pow(got2, zero);

		auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));
		copyroot = tenncor::sub(tenncor::pow(m, tenncor::div(got3, gotn1)), got2);
	}

	ead::NodeptrT<double> splitroot;
	{
		auto got1 = tenncor::cos(zero);
		auto got3 = tenncor::add(tenncor::add(one, zero), two);
		auto gotn1 = tenncor::sub(zero, one);
		auto got2 = tenncor::sub(two, zero);
		auto got22 = tenncor::max(two, zero);

		auto too = tenncor::div(got2, tenncor::mul(got1, got22));
		auto got11 = tenncor::eq(too, gotn1);

		splitroot = tenncor::mul(tenncor::mul(got11, got1), tenncor::mul(too, got3));
	}

	opt::OptCtx empty_rules = ead::parse<double>("");

	auto opted = opt::optimize({
		subroot->get_tensor(),
		root->get_tensor(),
		splitroot->get_tensor(),
		copyroot->get_tensor(),
	}, empty_rules);
	auto opt_subroot = opted[0];
	auto opt_root = opted[1];
	auto opt_splitroot = opted[2];
	auto opt_copyroot = opted[3];

	ASSERT_NE(nullptr, opt_subroot);
	ASSERT_NE(nullptr, opt_root);
	ASSERT_NE(nullptr, opt_splitroot);
	ASSERT_NE(nullptr, opt_copyroot);

	std::stringstream ss;
	CSVEquation ceq;
	opt_subroot->accept(ceq);
	opt_root->accept(ceq);
	opt_splitroot->accept(ceq);
	opt_copyroot->accept(ceq);
	ceq.to_stream(ss);

	std::list<std::string> expectlines =
	{
		"0:MUL,1:COS,0,white",
		"0:MUL,3:MAX,1,white",
		"10:ADD,0:MUL,1,white",
		"10:ADD,2:0,0,white",
		"11:POW,12:SUB,0,white",
		"11:POW,2:0,1,white",
		"12:SUB,2:0,1,white",
		"12:SUB,4:2,0,white",
		"13:DIV,14:ADD,0,white",
		"13:DIV,17:SUB,1,white",
		"14:ADD,15:ADD,0,white",
		"14:ADD,4:2,1,white",
		"15:ADD,16:1,0,white",
		"15:ADD,2:0,1,white",
		"17:SUB,16:1,1,white",
		"17:SUB,2:0,0,white",
		"18:MUL,19:MUL,0,white",
		"18:MUL,22:MUL,1,white",
		"19:MUL,1:COS,1,white",
		"19:MUL,20:EQ,0,white",
		"1:COS,2:0,0,white",
		"20:EQ,17:SUB,1,white",
		"20:EQ,21:DIV,0,white",
		"21:DIV,0:MUL,1,white",
		"21:DIV,12:SUB,0,white",
		"22:MUL,14:ADD,1,white",
		"22:MUL,21:DIV,0,white",
		"3:MAX,2:0,1,white",
		"3:MAX,4:2,0,white",
		"5:SUB,12:SUB,1,white",
		"5:SUB,6:POW,0,white",
		"6:POW,13:DIV,1,white",
		"6:POW,7:MIN,0,white",
		"7:MIN,8:MIN,0,white",
		"7:MIN,9:MIN,1,white",
		"8:MIN,1:COS,1,white",
		"8:MIN,3:MAX,0,white",
		"9:MIN,10:ADD,0,white",
		"9:MIN,11:POW,1,white",
	};
	expectlines.sort();
	std::list<std::string> gotlines;
	std::string line;
	while (std::getline(ss, line))
	{
		gotlines.push_back(line);
	}
	gotlines.sort();
	std::vector<std::string> diffs;
	std::set_symmetric_difference(expectlines.begin(), expectlines.end(),
		gotlines.begin(), gotlines.end(), std::back_inserter(diffs));

	std::stringstream diffstr;
	for (auto diff : diffs)
	{
		diffstr << diff << "\n";
	}
	std::string diffmsg = diffstr.str();
	EXPECT_EQ(0, diffmsg.size()) << "mismatching edges:\n" << diffmsg;
}


#endif // DISABLE_OPTIMIZATION_TEST
