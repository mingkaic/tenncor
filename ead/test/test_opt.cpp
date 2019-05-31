
#ifndef DISABLE_OPTIMIZATION_TEST

#include "gtest/gtest.h"

#include "dbg/stream/ade_csv.hpp"

#include "testutil/common.hpp"

#include "ead/opt/ops_prune.hpp"

#include "ead/opt/conversion.hpp"
#include "ead/opt/parse.hpp"

#include "ead/generated/api.hpp"

#include "ead/constant.hpp"


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
	ead::opt::ConversionsT<double> rules = {
		std::make_shared<ead::opt::CalcConstantConversion<double>>()
	};

	auto vfunc = age::sin(var);
	{
		ead::NodesT<double> opts = {vfunc};
		ead::opt::optimize<double>(opts, rules);
		auto opt_vfunc = opts[0];
		// expect vfunc to equal opt_vfunc
		std::stringstream ss;
		ss <<
			"(SIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, opt_vfunc->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	auto cfunc = age::sin(two);
	{
		ead::NodesT<double> opts = {cfunc};
		ead::opt::optimize<double>(opts, rules);
		auto opt_cfunc = opts[0];
		// expect sin(2) to equal opt_cfunc
		std::stringstream ss;
		ss <<
			"(" << std::sin(2) << "[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, opt_cfunc->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	auto adv_func = age::mul(age::add(vfunc, cfunc), age::pow(three, four));
	{
		ead::NodesT<double> opts = {adv_func};
		ead::opt::optimize<double>(opts, rules);
		auto opt_adv_func = opts[0];
		// expect (sin(var) + sin(2)) * 81 to equal opt_adv_func
		std::stringstream ss;
		ss <<
			"(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(ADD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   `--(SIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   |   `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
			" |   `--(" << std::sin(2) << "[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(" << std::pow(3, 4) << "[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, opt_adv_func->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}
}


TEST(OPTIMIZE, PruneSingleZeros)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"special_var"));

	ead::NodeptrT<double> zero =
		ead::make_constant_scalar<double>(0, ade::Shape());
	auto rules = ead::opt::get_configs<double>();

	auto wunfunc = age::pow(var, zero);
	auto zrofunc = age::pow(zero, var);
	{
		ead::NodesT<double> opts = {wunfunc, zrofunc};
		ead::opt::optimize<double>(opts, rules);
		auto opt_wunfunc = opts[0];
		auto opt_zrofunc = opts[1];
		// expect both opt_wunfunc to be 1
		std::istringstream ss("(1[1\\1\\1\\1\\1\\1\\1\\1])");
		auto compare_str = compare_graph(ss, opt_wunfunc->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;

		// expect both opt_zrofunc to be 0
		std::istringstream ss2("(0[1\\1\\1\\1\\1\\1\\1\\1])");
		auto compare_str2 = compare_graph(ss2, opt_zrofunc->get_tensor());
		EXPECT_EQ(0, compare_str2.size()) << compare_str2;
	}

	auto lvfunc = age::add(var, zero);
	auto rvfunc = age::add(zero, var);
	{
		ead::NodesT<double> opts = {lvfunc, rvfunc};
		ead::opt::optimize<double>(opts, rules);
		auto opt_lvfunc = opts[0];
		auto opt_rvfunc = opts[1];
		// expect both vfuncs to equal var
		std::string expect = "(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";

		std::istringstream ss(expect);
		auto compare_str = compare_graph(ss, opt_lvfunc->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;

		std::istringstream ss2(expect);
		auto compare_str2 = compare_graph(ss2, opt_rvfunc->get_tensor());
		EXPECT_EQ(0, compare_str2.size()) << compare_str2;
	}

	auto lzero = age::mul(var, zero);
	auto rzero = age::mul(zero, var);
	{
		ead::NodesT<double> opts = {lzero, rzero};
		ead::opt::optimize<double>(opts, rules);
		auto opt_lzero = opts[0];
		auto opt_rzero = opts[1];
		// expect both zeros to equal 0
		std::string expect = "(0[1\\1\\1\\1\\1\\1\\1\\1])";

		std::istringstream ss(expect);
		auto compare_str = compare_graph(ss, opt_lzero->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;

		std::istringstream ss2(expect);
		auto compare_str2 = compare_graph(ss2, opt_rzero->get_tensor());
		EXPECT_EQ(0, compare_str2.size()) << compare_str2;
	}

	auto posvar = age::sub(var, zero);
	auto negvar = age::sub(zero, var);
	{
		ead::NodesT<double> opts = {posvar, negvar};
		ead::opt::optimize<double>(opts, rules);
		auto opt_posvar = opts[0];
		auto opt_negvar = opts[1];
		// expect opt_posvar to equal var
		std::istringstream ss("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])");
		auto compare_str = compare_graph(ss, opt_posvar->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;

		// expect opt_negvar equal -var
		std::stringstream ss2;
		ss2 <<
			"(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str2 = compare_graph(ss2, opt_negvar->get_tensor());
		EXPECT_EQ(0, compare_str2.size()) << compare_str2;
	}

	auto divz = age::div(zero, var);
	{
		ead::NodesT<double> opts = {divz};
		ead::opt::optimize<double>(opts, rules);
		auto opt_divz = opts[0];
		// expect opt_divz to equal zero
		std::istringstream ss("(0[1\\1\\1\\1\\1\\1\\1\\1])");
		auto compare_str = compare_graph(ss, opt_divz->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	auto no_opt = age::max(zero, var);
	{
		ead::NodesT<double> opts = {no_opt};
		ead::opt::optimize<double>(opts, rules);
		auto not_opt = opts[0];
		// expect opt_divz to equal zero
		std::stringstream ss;
		ss <<
			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, not_opt->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}
}


TEST(OPTIMIZE, PruneZeroGraph)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"var"));
	ead::NodeptrT<double> var2 = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"var2"));

	ead::NodeptrT<double> zero =
		ead::make_constant_scalar<double>(0, ade::Shape());
	auto rules = ead::opt::get_configs<double>();

	auto got1 = age::cos(zero);
	auto got3 = age::add(zero, var2);
	auto gotn1 = age::sub(zero, var);
	auto got2 = age::sub(var2, zero);
	auto got22 = age::max(var2, zero);

	auto too = age::add(zero, age::mul(got1, got22));
	auto got11 = age::pow(got2, zero);

	auto m = age::min(age::max(got22, got1), age::lt(too, got11));
	auto nocascades = age::sub(age::pow(m, age::div(got3, gotn1)), got2);

	ead::NodesT<double> opts = {nocascades};
	ead::opt::optimize<double>(opts, rules);
	auto opt_nocascades = opts[0];
	std::stringstream ss;
	ss <<
		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   |   `--(variable:var2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   |   `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(LT[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       |   `--(variable:var2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       |   `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |       `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(variable:var2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(variable:var2[1\\1\\1\\1\\1\\1\\1\\1])";
	auto compare_str = compare_graph(ss, opt_nocascades->get_tensor());
	EXPECT_EQ(0, compare_str.size()) << compare_str;

	auto got0 = age::tan(zero);
	opts = {age::pow(nocascades, got0)};
	ead::opt::optimize<double>(opts, rules);
	auto opt_cascades = opts[0];
	EXPECT_STREQ("1",
		opt_cascades->get_tensor()->to_string().c_str());
}


TEST(OPTIMIZE, PruneSingleOnes)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"special_var"));

	ead::NodeptrT<double> one =
		ead::make_constant_scalar<double>(1, ade::Shape());
	auto rules = ead::opt::get_configs<double>();

	auto vfunc = age::pow(var, one);
	auto wunfunc = age::pow(one, var);
	{
		ead::NodesT<double> opts = {vfunc, wunfunc};
		ead::opt::optimize<double>(opts, rules);
		auto opt_vfunc = opts[0];
		auto opt_wunfunc = opts[1];
		// expect both opt_vfunc to be var
		std::istringstream ss("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])");
		auto compare_str = compare_graph(ss, opt_vfunc->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;

		// expect both opt_wunfunc to be 1
		std::istringstream ss2("(1[1\\1\\1\\1\\1\\1\\1\\1])");
		auto compare_str2 = compare_graph(ss2, opt_wunfunc->get_tensor());
		EXPECT_EQ(0, compare_str2.size()) << compare_str2;
	}

	auto lvfunc = age::mul(var, one);
	auto rvfunc = age::mul(one, var);
	{
		ead::NodesT<double> opts = {lvfunc, rvfunc};
		ead::opt::optimize<double>(opts, rules);
		auto opt_lvfunc = opts[0];
		auto opt_rvfunc = opts[1];
		// expect both vfuncs to equal var
		std::string expect = "(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";

		std::istringstream ss(expect);
		auto compare_str = compare_graph(ss, opt_lvfunc->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;

		std::istringstream ss2(expect);
		auto compare_str2 = compare_graph(ss2, opt_rvfunc->get_tensor());
		EXPECT_EQ(0, compare_str2.size()) << compare_str2;
	}

	auto nomer = age::div(var, one);
	{
		ead::NodesT<double> opts = {nomer};
		ead::opt::optimize<double>(opts, rules);
		auto opt_nomer = opts[0];
		// expect opt_divz to equal zero
		std::istringstream ss("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])");
		auto compare_str = compare_graph(ss, opt_nomer->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	auto wun = age::div(var, var);
	{
		ead::NodesT<double> opts = {wun};
		ead::opt::optimize<double>(opts, rules);
		auto opt_wun = opts[0];
		std::istringstream ss("(1[1\\1\\1\\1\\1\\1\\1\\1])");
		auto compare_str = compare_graph(ss, opt_wun->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	auto no_opt = age::max(one, var);
	{
		ead::NodesT<double> opts = {no_opt};
		ead::opt::optimize<double>(opts, rules);
		auto not_opt = opts[0];
		// expect opt_divz to equal zero
		std::stringstream ss;
		ss <<
			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";
		auto compare_str = compare_graph(ss, not_opt->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}
}


TEST(OPTIMIZE, PruneOneGraph)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"var"));

	ead::NodeptrT<double> one =
		ead::make_constant_scalar<double>(1, ade::Shape());
	auto rules = ead::opt::get_configs<double>();

	auto got0 = age::log(one);
	auto got1 = age::sqrt(one);
	auto got3 = age::mul(one, var);
	auto got00 = age::pow(one, var);
	auto got = age::max(var, one);

	auto too = age::add(got1, age::mul(got0, got00));
	auto got11 = age::pow(var, one);

	auto m = age::min(age::max(got1, too), got11);
	auto root = age::sub(age::pow(m, age::div(got3, got)), var);

	ead::NodesT<double> opts = {root};
	ead::opt::optimize<double>(opts, rules);
	auto opt = opts[0];
	std::stringstream ss;
	ss <<
		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   |   `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" |           `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
		" `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])";
	auto compare_str = compare_graph(ss, opt->get_tensor());
	EXPECT_EQ(0, compare_str.size()) << compare_str;
}

// TEST(OPTIMIZATION, ops_prune_singles)
// {
// 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());
// 	ead::NodeptrT<double> three = ead::make_constant_scalar<double>(3, ade::Shape());

// 	// merge redundent double reduced argument
// 	auto got0 = ead::ops_prune({age::reduce_sum(age::reduce_sum(zero))})[0];
// 	{
// 		std::stringstream ss;
// 		ss <<
// 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			" `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n";
// 		auto compare_str = compare_graph(ss, got0);
// 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// 	}

// 	// don't merge non-redundent double reduced argument
// 	auto got_0 = ead::ops_prune({age::reduce_sum(age::reduce_sum(zero, 1), 0)})[0];
// 	{
// 		std::stringstream ss;
// 		ss <<
// 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			" `--(add[3\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			"     `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n";
// 		auto compare_str = compare_graph(ss, got_0);
// 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// 	}

// 	// don't merge mul-reduced_add
// 	auto got_0_1 = ead::ops_prune({age::mul({age::reduce_sum(zero), one})})[0];
// 	{
// 		std::stringstream ss;
// 		ss <<
// 			"(mul[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			" `--(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			" |   `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			" `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n";
// 		auto compare_str = compare_graph(ss, got_0_1);
// 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// 	}
// }


// TEST(OPTIMIZATION, ops_prune_graph)
// {
// 	ead::NodeptrT<double> zero(ead::Variable<double>::get(ade::Shape({3, 4}), "0"));
// 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());
// 	ead::NodeptrT<double> three = ead::make_constant_scalar<double>(3, ade::Shape());

// 	auto got1 = age::cos(three);
// 	auto got3 = age::mul({one, three, two});
// 	auto gotn1 = age::sub(three, one);
// 	auto got2 = age::sub(two, three);
// 	auto got22 = age::min({two, three});

// 	auto too = age::mul(age::reduce_mul(age::reduce_mul_1d(zero, 0), 0),
// 		age::reduce_mul(age::mul({got1, got22})));
// 	auto got11 = age::pow(got2, three);

// 	auto m = age::min({got22, got1, too, got11});
// 	auto root = ead::ops_prune({age::sub(
// 		age::min({m, age::div(got3, gotn1)}), got2)})[0];

// 	std::stringstream ss;
// 	ss <<
// 		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   `--(mul[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   `--(mul[4\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   |   `--(0[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |       `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |       `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |       `--(mul[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |       |   `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |       |   `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |       |   `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |       `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |           `--(3[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |           `--(1[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		"     `--(2[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		"     `--(3[1\\1\\1\\1\\1\\1\\1\\1])";
// 	auto compare_str = compare_graph(ss, root);
// 	EXPECT_EQ(0, compare_str.size()) << compare_str;
// }


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
		auto got1 = age::cos(zero);
		auto got3 = age::add(age::add(one, zero), two);
		auto gotn1 = age::sub(zero, one);
		auto got2 = age::sub(two, zero);
		auto got22 = age::max(two, zero);

		auto too = age::add(zero, age::mul(got1, got22));
		auto got11 = age::pow(got2, zero);

		auto m = age::min(age::min(got22, got1), age::min(too, got11));
		root = age::sub(age::pow(m, age::div(got3, gotn1)), got2);
	}

	ead::NodeptrT<double> subroot;
	{
		auto other_got1 = age::cos(zero);
		auto got22 = age::max(two, zero);
		subroot = age::mul(other_got1, got22);
	}

	ead::NodeptrT<double> copyroot;
	{
		auto got1 = age::cos(zero);
		auto got3 = age::add(age::add(one, zero), two);
		auto gotn1 = age::sub(zero, one);
		auto got2 = age::sub(two, zero);
		auto got22 = age::max(two, zero);

		auto too = age::add(zero, age::mul(got1, got22));
		auto got11 = age::pow(got2, zero);

		auto m = age::min(age::min(got22, got1), age::min(too, got11));
		copyroot = age::sub(age::pow(m, age::div(got3, gotn1)), got2);
	}

	ead::NodeptrT<double> splitroot;
	{
		auto got1 = age::cos(zero);
		auto got3 = age::add(age::add(one, zero), two);
		auto gotn1 = age::sub(zero, one);
		auto got2 = age::sub(two, zero);
		auto got22 = age::max(two, zero);

		auto too = age::div(got2, age::mul(got1, got22));
		auto got11 = age::eq(too, gotn1);

		splitroot = age::mul(age::mul(got11, got1), age::mul(too, got3));
	}

	ead::opt::ConversionsT<double> empty_rules;
	ead::NodesT<double> opts = {subroot, root, splitroot, copyroot};
	ead::opt::optimize(opts, empty_rules);
	auto opt_subroot = opts[0];
	auto opt_root = opts[1];
	auto opt_splitroot = opts[2];
	auto opt_copyroot = opts[3];

	ASSERT_NE(nullptr, opt_subroot);
	ASSERT_NE(nullptr, opt_root);
	ASSERT_NE(nullptr, opt_splitroot);
	ASSERT_NE(nullptr, opt_copyroot);

	ASSERT_NE(nullptr, opt_subroot->get_tensor());
	ASSERT_NE(nullptr, opt_root->get_tensor());
	ASSERT_NE(nullptr, opt_splitroot->get_tensor());
	ASSERT_NE(nullptr, opt_copyroot->get_tensor());

	std::stringstream ss;
	CSVEquation ceq;
	opt_subroot->get_tensor()->accept(ceq);
	opt_root->get_tensor()->accept(ceq);
	opt_splitroot->get_tensor()->accept(ceq);
	opt_copyroot->get_tensor()->accept(ceq);
	ceq.to_stream(ss);

	std::list<std::string> expectlines =
	{
		"0:MUL,1:COS,0,white",
		"1:COS,2:variable:0,0,white",
		"0:MUL,3:MAX,1,white",
		"3:MAX,4:variable:2,0,white",
		"3:MAX,2:variable:0,1,white",
		"5:SUB,6:POW,0,white",
		"6:POW,7:MIN,0,white",
		"7:MIN,8:MIN,0,white",
		"8:MIN,3:MAX,0,white",
		"8:MIN,1:COS,1,white",
		"7:MIN,9:MIN,1,white",
		"9:MIN,10:ADD,0,white",
		"10:ADD,2:variable:0,0,white",
		"10:ADD,0:MUL,1,white",
		"9:MIN,11:POW,1,white",
		"11:POW,12:SUB,0,white",
		"12:SUB,4:variable:2,0,white",
		"12:SUB,2:variable:0,1,white",
		"11:POW,2:variable:0,1,white",
		"6:POW,13:DIV,1,white",
		"13:DIV,14:ADD,0,white",
		"14:ADD,15:ADD,0,white",
		"15:ADD,16:variable:1,0,white",
		"15:ADD,2:variable:0,1,white",
		"14:ADD,4:variable:2,1,white",
		"13:DIV,17:SUB,1,white",
		"17:SUB,2:variable:0,0,white",
		"17:SUB,16:variable:1,1,white",
		"5:SUB,12:SUB,1,white",
		"18:MUL,19:MUL,0,white",
		"19:MUL,20:EQ,0,white",
		"20:EQ,21:DIV,0,white",
		"21:DIV,12:SUB,0,white",
		"21:DIV,0:MUL,1,white",
		"20:EQ,17:SUB,1,white",
		"19:MUL,1:COS,1,white",
		"18:MUL,22:MUL,1,white",
		"22:MUL,21:DIV,0,white",
		"22:MUL,14:ADD,1,white",
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
