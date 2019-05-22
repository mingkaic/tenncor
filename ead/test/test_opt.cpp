
#ifndef DISABLE_OPTIMIZATION_TEST

#include "gtest/gtest.h"

#include "dbg/stream/ade_csv.hpp"

#include "testutil/common.hpp"

// #include "ead/opt/zero_prune.hpp"
// #include "ead/opt/one_prune.hpp"
// #include "ead/opt/const_merge.hpp"
// #include "ead/opt/ops_reuse.hpp"
#include "ead/opt/ops_prune.hpp"

#include "ead/opt/conversion.hpp"
#include "ead/opt/parse.hpp"

#include "ead/generated/api.hpp"

#include "ead/constant.hpp"


TEST(OPTIMIZE, PreCalcConstants)
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

	auto vfunc = age::sin(var);
	{
		ead::opt::Representer<double> repr;
		vfunc->get_tensor()->accept(repr);
		auto opt_vfunc = ead::opt::unrepresent(repr, {vfunc})[0];
		// expect vfunc to equal opt_vfunc
		std::stringstream ss;
		ss <<
			"(SIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(special_var([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, opt_vfunc->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	auto cfunc = age::sin(two);
	{
		ead::opt::Representer<double> repr;
		cfunc->get_tensor()->accept(repr);
		auto opt_cfunc = ead::opt::unrepresent(repr, {cfunc})[0];
		// expect sin(2) to equal opt_cfunc
		std::stringstream ss;
		ss <<
			"(" << std::sin(2) << "([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, opt_cfunc->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}

	auto adv_func = age::mul(age::add(vfunc, cfunc), age::pow(three, four));
	{
		ead::opt::Representer<double> repr;
		adv_func->get_tensor()->accept(repr);
		auto opt_adv_func = ead::opt::unrepresent(repr, {adv_func})[0];
		// expect (sin(var) + sin(2)) * 81 to equal opt_adv_func
		std::stringstream ss;
		ss <<
			"(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(ADD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   `--(SIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" |   |   `--(special_var([1\\1\\1\\1\\1\\1\\1\\1]))\n"
			" |   `--(" << std::sin(2) << "([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
			" `--(" << std::pow(3, 4) << "([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str = compare_graph(ss, opt_adv_func->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}
}


TEST(OPTIMIZE, PruneSinglesZeros)
{
	ead::NodeptrT<double> var = ead::convert_to_node(
		ead::make_variable_scalar<double>(0, ade::Shape(),
		"special_var"));

	ead::NodeptrT<double> zero =
		ead::make_constant_scalar<double>(0, ade::Shape());
	auto rules = ead::opt::get_configs<double>();

	auto lvfunc = age::add(var, zero);
	auto rvfunc = age::add(zero, var);
	{
		ead::NodesT<double> opts = {lvfunc, rvfunc};
		ead::opt::optimize<double>(opts, rules);
		auto opt_lvfunc = opts[0];
		auto opt_rvfunc = opts[1];
		// expect both vfuncs to equal var
		std::string expect = "(special_var([1\\1\\1\\1\\1\\1\\1\\1]))";

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
		std::string expect = "(0([1\\1\\1\\1\\1\\1\\1\\1]))";

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
		std::istringstream ss("(special_var([1\\1\\1\\1\\1\\1\\1\\1]))");
		auto compare_str = compare_graph(ss, opt_posvar->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;

		// expect opt_negvar equal -var
		std::stringstream ss2;
		ss2 <<
			"(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
			" `--(special_var([1\\1\\1\\1\\1\\1\\1\\1]))";
		auto compare_str2 = compare_graph(ss2, opt_negvar->get_tensor());
		EXPECT_EQ(0, compare_str2.size()) << compare_str2;
	}

	auto divz = age::div(zero, var);
	{
		ead::NodesT<double> opts = {divz};
		ead::opt::optimize<double>(opts, rules);
		auto opt_divz = opts[0];
		// expect opt_divz to equal zero
		std::istringstream ss("(0([1\\1\\1\\1\\1\\1\\1\\1]))");
		auto compare_str = compare_graph(ss, opt_divz->get_tensor());
		EXPECT_EQ(0, compare_str.size()) << compare_str;
	}
}


// TEST(OPTIMIZATION, zero_prune_singles)
// {
// 	ead::NodeptrT<double> zero = ead::make_constant_scalar<double>(0, ade::Shape());
// 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());

// 	auto got0 = ead::zero_prune<double>({age::sin(zero)})[0];
// 	EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got0->get_tensor()->to_string().c_str());

// 	auto got1 = ead::zero_prune<double>({age::cos(zero)})[0];
// 	EXPECT_STREQ("1([1\\1\\1\\1\\1\\1\\1\\1])", got1->get_tensor()->to_string().c_str());

// 	auto gottwo = ead::zero_prune<double>({age::add(zero, two)})[0];
// 	EXPECT_STREQ("2([1\\1\\1\\1\\1\\1\\1\\1])", gottwo->get_tensor()->to_string().c_str());

// 	auto gotn1 = ead::zero_prune<double>({age::sub(zero, one)})[0];
// 	{
// 		std::stringstream ss;
// 		ss <<
// 			"(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))";
// 		auto compare_str = compare_graph(ss, gotn1->get_tensor());
// 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// 	}

// 	auto got2 = ead::zero_prune<double>({age::sub(two, zero)})[0];
// 	EXPECT_STREQ("2([1\\1\\1\\1\\1\\1\\1\\1])", got2->get_tensor()->to_string().c_str());

// 	auto got00 = ead::zero_prune<double>({age::mul(two, zero)})[0];
// 	EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got00->get_tensor()->to_string().c_str());

// 	auto got000 = ead::zero_prune<double>({age::div(zero, two)})[0];
// 	EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got000->get_tensor()->to_string().c_str());

// 	EXPECT_FATAL(ead::zero_prune<double>({age::div(one, zero)}), "cannot DIV by zero");

// 	auto gotnormal = ead::zero_prune<double>({age::max(two, zero)})[0];
// 	{
// 		std::stringstream ss;
// 		ss <<
// 			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 			" `--(0([1\\1\\1\\1\\1\\1\\1\\1]))";
// 		auto compare_str = compare_graph(ss, gotnormal->get_tensor());
// 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// 	}
// }


// TEST(OPTIMIZATION, zero_prune_graph)
// {
// 	ead::NodeptrT<double> zero = ead::make_constant_scalar<double>(0, ade::Shape());
// 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());

// 	auto got1 = age::cos(zero);
// 	auto got3 = age::add(zero, two);
// 	auto gotn1 = age::sub(zero, one);
// 	auto got2 = age::sub(two, zero);
// 	auto got22 = age::max(two, zero);

// 	auto too = age::add(zero, age::mul(got1, got22));
// 	auto got11 = age::pow(got2, zero);

// 	auto m = age::min(age::max(got22, got1), age::lt(too, got11));
// 	auto nocascades = age::sub(age::pow(m, age::div(got3, gotn1)), got2);

// 	auto opt_nocascades = ead::zero_prune<double>({nocascades})[0];
// 	std::stringstream ss;
// 	ss <<
// 		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   |   |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |   |   |   `--(0([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |   |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |   `--(LT[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |       `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |       |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |       |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |       |       `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |       |       `--(0([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |       `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |       `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |       `--(NEG[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |           `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))";
// 	auto compare_str = compare_graph(ss, opt_nocascades->get_tensor());
// 	EXPECT_EQ(0, compare_str.size()) << compare_str;

// 	auto got0 = age::tan(zero);
// 	auto opt_cascades = ead::zero_prune<double>({age::pow(nocascades, got0)})[0];
// 	EXPECT_STREQ("1([1\\1\\1\\1\\1\\1\\1\\1])",
// 		opt_cascades->get_tensor()->to_string().c_str());
// }


// TEST(OPTIMIZATION, one_prune_singles)
// {
// 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());

// 	auto got0 = ead::one_prune<double>({age::log(one)})[0];
// 	EXPECT_STREQ("0([1\\1\\1\\1\\1\\1\\1\\1])", got0->get_tensor()->to_string().c_str());

// 	auto got1 = ead::one_prune<double>({age::sqrt(one)})[0];
// 	EXPECT_STREQ("1([1\\1\\1\\1\\1\\1\\1\\1])", got1->get_tensor()->to_string().c_str());

// 	auto got02 = ead::one_prune<double>({age::mul(one, two)})[0];
// 	EXPECT_STREQ("2([1\\1\\1\\1\\1\\1\\1\\1])", got02->get_tensor()->to_string().c_str());

// 	auto got2 = ead::one_prune<double>({age::div(two, one)})[0];
// 	EXPECT_STREQ("2([1\\1\\1\\1\\1\\1\\1\\1])", got2->get_tensor()->to_string().c_str());

// 	auto gottoo = ead::one_prune<double>({age::pow(two, one)})[0];
// 	EXPECT_STREQ("2([1\\1\\1\\1\\1\\1\\1\\1])", gottoo->get_tensor()->to_string().c_str());

// 	auto gotone = ead::one_prune<double>({age::pow(one, two)})[0];
// 	EXPECT_STREQ("1([1\\1\\1\\1\\1\\1\\1\\1])", gotone->get_tensor()->to_string().c_str());

// 	auto gotnormal = ead::one_prune<double>({age::max(two, one)})[0];
// 	{
// 		std::stringstream ss;
// 		ss <<
// 			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))";
// 		auto compare_str = compare_graph(ss, gotnormal->get_tensor());
// 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// 	}
// }


// TEST(OPTIMIZATION, one_prune_graph)
// {
// 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());

// 	auto got0 = age::log(one);
// 	auto got1 = age::sqrt(one);
// 	auto got3 = age::mul(one, two);
// 	auto got00 = age::pow(one, two);
// 	auto got = age::max(two, one);

// 	auto too = age::add(got1, age::mul(got0, got00));
// 	auto got11 = age::pow(two, one);

// 	auto m = age::min(age::max(got1, too), got11);
// 	auto root = age::sub(age::pow(m, age::div(got3, got)), two);

// 	auto opt = ead::one_prune<double>({root})[0];
// 	std::stringstream ss;
// 	ss <<
// 		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |   |   `--(ADD[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |   |   |       `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |   |       `--(0([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |       `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// 		" |           `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" |           `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// 		" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))";
// 	auto compare_str = compare_graph(ss, opt->get_tensor());
// 	EXPECT_EQ(0, compare_str.size()) << compare_str;
// }

// // TEST(OPTIMIZATION, ops_prune_singles)
// // {
// // 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// // 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());
// // 	ead::NodeptrT<double> three = ead::make_constant_scalar<double>(3, ade::Shape());

// // 	// merge same consecutive nnary
// // 	auto got1123 = ead::ops_prune({age::add({one, age::add(one, two), three})})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
// // 		auto compare_str = compare_graph(ss, got1123);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}

// // 	// don't merge different nnary
// // 	auto got1_12_3 = ead::ops_prune({age::add({one, age::max({one, two}), three})})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
// // 		auto compare_str = compare_graph(ss, got1_12_3);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}

// // 	// merge single unary argument of nnary
// // 	auto got213 = ead::ops_prune({age::add({two, age::max({one}), three})})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
// // 		auto compare_str = compare_graph(ss, got213);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}

// // 	// don't merge single unary argument of non-nnary
// // 	auto got2_1_3 = ead::ops_prune({age::add({two, age::tan(one), three})})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(TAN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
// // 		auto compare_str = compare_graph(ss, got2_1_3);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}

// // 	ead::NodeptrT<double> zero(ead::Variable<double>::get(ade::Shape({3, 4}), "0"));
// // 	// merge reduced argument
// // 	auto got2103 = ead::ops_prune({age::add({two, one, age::reduce_add(zero), three})})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
// // 		auto compare_str = compare_graph(ss, got2103);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}

// // 	// merge reduced add
// // 	ead::NodeptrT<double> shaped_one = ead::make_constant_scalar<double>(1, ade::Shape({3, 4})));
// // 	auto got10 = ead::ops_prune({age::reduce_add(age::add({shaped_one, zero}))})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(1([3\\4\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(0([3\\4\\1\\1\\1\\1\\1\\1]))";
// // 		auto compare_str = compare_graph(ss, got10);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}

// // 	// merge redundent double reduced argument
// // 	auto got0 = ead::ops_prune({age::reduce_add(age::reduce_add(zero))})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n";
// // 		auto compare_str = compare_graph(ss, got0);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}

// // 	// don't merge non-redundent double reduced argument
// // 	auto got_0 = ead::ops_prune({age::reduce_add(age::reduce_add(zero, 1), 0)})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(add[3\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			"     `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n";
// // 		auto compare_str = compare_graph(ss, got_0);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}

// // 	// don't merge mul-reduced_add
// // 	auto got_0_1 = ead::ops_prune({age::mul({age::reduce_add(zero), one})})[0];
// // 	{
// // 		std::stringstream ss;
// // 		ss <<
// // 			"(mul[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" `--(add[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 			" |   `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n" <<
// // 			" `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n";
// // 		auto compare_str = compare_graph(ss, got_0_1);
// // 		EXPECT_EQ(0, compare_str.size()) << compare_str;
// // 	}
// // }


// // TEST(OPTIMIZATION, ops_prune_graph)
// // {
// // 	ead::NodeptrT<double> zero(ead::Variable<double>::get(ade::Shape({3, 4}), "0"));
// // 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// // 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());
// // 	ead::NodeptrT<double> three = ead::make_constant_scalar<double>(3, ade::Shape());

// // 	auto got1 = age::cos(three);
// // 	auto got3 = age::mul({one, three, two});
// // 	auto gotn1 = age::sub(three, one);
// // 	auto got2 = age::sub(two, three);
// // 	auto got22 = age::min({two, three});

// // 	auto too = age::mul(age::reduce_mul(age::reduce_mul_1d(zero, 0), 0),
// // 		age::reduce_mul(age::mul({got1, got22})));
// // 	auto got11 = age::pow(got2, three);

// // 	auto m = age::min({got22, got1, too, got11});
// // 	auto root = ead::ops_prune({age::sub(
// // 		age::min({m, age::div(got3, gotn1)}), got2)})[0];

// // 	std::stringstream ss;
// // 	ss <<
// // 		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |   |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   `--(mul[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |   |   `--(mul[4\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |   |   |   `--(0([3\\4\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |   |   |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |   |       `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   |       `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |   |   `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |   |   |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   |   |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |       `--(mul[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |       |   `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |       |   `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |       |   `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |       `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		" |           `--(3([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" |           `--(1([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		" `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n" <<
// // 		"     `--(2([1\\1\\1\\1\\1\\1\\1\\1]))\n" <<
// // 		"     `--(3([1\\1\\1\\1\\1\\1\\1\\1]))";
// // 	auto compare_str = compare_graph(ss, root);
// // 	EXPECT_EQ(0, compare_str.size()) << compare_str;
// // }


// TEST(OPTIMIZATION, const_merge_graph)
// {
// 	ead::NodeptrT<double> zero = ead::make_constant_scalar<double>(0, ade::Shape());
// 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());

// 	auto got1 = age::cos(zero);
// 	auto got3 = age::add(one, age::add(zero, two));
// 	auto gotn1 = age::sub(zero, one);
// 	auto got2 = age::sub(two, zero);
// 	auto got22 = age::max(two, zero);

// 	auto too = age::add(zero, age::mul(got1, got22));
// 	auto got11 = age::pow(got2, zero);

// 	auto m = age::min(age::min(got22, got1), age::min(too, got11));
// 	auto root = age::sub(age::pow(m, age::div(got3, gotn1)), got2);

// 	auto opt = ead::const_merge<double>({root})[0];
// 	EXPECT_STREQ("-1([1\\1\\1\\1\\1\\1\\1\\1])", opt->get_tensor()->to_string().c_str());
// }


// TEST(OPTIMIZATION, reuse_op_graph)
// {
// 	ead::NodeptrT<double> zero = ead::make_constant_scalar<double>(0, ade::Shape());
// 	ead::NodeptrT<double> zero2 = ead::make_constant_scalar<double>(0, ade::Shape());
// 	ead::NodeptrT<double> zero3 = ead::make_constant_scalar<double>(0, ade::Shape());
// 	ead::NodeptrT<double> one = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> one2 = ead::make_constant_scalar<double>(1, ade::Shape());
// 	ead::NodeptrT<double> two = ead::make_constant_scalar<double>(2, ade::Shape());
// 	ead::NodeptrT<double> two2 = ead::make_constant_scalar<double>(2, ade::Shape());

// 	ead::NodeptrT<double> root;
// 	{
// 		auto got1 = age::cos(zero);
// 		auto got3 = age::add(age::add(one, zero), two2);
// 		auto gotn1 = age::sub(zero2, one2);
// 		auto got2 = age::sub(two, zero3);
// 		auto got22 = age::max(two, zero2);

// 		auto too = age::add(zero, age::mul(got1, got22));
// 		auto got11 = age::pow(got2, zero3);

// 		auto m = age::min(age::min(got22, got1), age::min(too, got11));
// 		root = age::sub(age::pow(m, age::div(got3, gotn1)), got2);
// 	}

// 	ead::NodeptrT<double> subroot;
// 	{
// 		auto other_got1 = age::cos(zero);
// 		auto got22 = age::max(two2, zero3);
// 		subroot = age::mul(other_got1, got22);
// 	}

// 	ead::NodeptrT<double> copyroot;
// 	{
// 		auto got1 = age::cos(zero);
// 		auto got3 = age::add(age::add(one, zero), two2);
// 		auto gotn1 = age::sub(zero2, one2);
// 		auto got2 = age::sub(two, zero3);
// 		auto got22 = age::max(two, zero2);

// 		auto too = age::add(zero, age::mul(got1, got22));
// 		auto got11 = age::pow(got2, zero3);

// 		auto m = age::min(age::min(got22, got1), age::min(too, got11));
// 		copyroot = age::sub(age::pow(m, age::div(got3, gotn1)), got2);
// 	}

// 	ead::NodeptrT<double> splitroot;
// 	{
// 		auto got1 = age::cos(zero);
// 		auto got3 = age::add(age::add(one, zero), two2);
// 		auto gotn1 = age::sub(zero2, one2);
// 		auto got2 = age::sub(two, zero3);
// 		auto got22 = age::max(two, zero2);

// 		auto too = age::div(got2, age::mul(got1, got22));
// 		auto got11 = age::eq(too, gotn1);

// 		splitroot = age::mul(age::mul(got11, got1), age::mul(too, got3));
// 	}

// 	auto opts = ead::ops_reuse<double>({subroot, root, splitroot, copyroot});
// 	auto opt_subroot = opts[0];
// 	auto opt_root = opts[1];
// 	auto opt_splitroot = opts[2];
// 	auto opt_copyroot = opts[3];

// 	ASSERT_NE(nullptr, opt_subroot);
// 	ASSERT_NE(nullptr, opt_root);
// 	ASSERT_NE(nullptr, opt_splitroot);
// 	ASSERT_NE(nullptr, opt_copyroot);

// 	ASSERT_NE(nullptr, opt_subroot->get_tensor());
// 	ASSERT_NE(nullptr, opt_root->get_tensor());
// 	ASSERT_NE(nullptr, opt_splitroot->get_tensor());
// 	ASSERT_NE(nullptr, opt_copyroot->get_tensor());

// 	std::stringstream ss;
// 	CSVEquation ceq;
// 	opt_subroot->get_tensor()->accept(ceq);
// 	opt_root->get_tensor()->accept(ceq);
// 	opt_splitroot->get_tensor()->accept(ceq);
// 	opt_copyroot->get_tensor()->accept(ceq);
// 	ceq.to_stream(ss);

// 	std::list<std::string> expectlines =
// 	{
// 		"0:MUL,1:COS,0,white",
// 		"1:COS,2:0([1\\1\\1\\1\\1\\1\\1\\1]),0,white",
// 		"0:MUL,3:MAX,1,white",
// 		"3:MAX,4:2([1\\1\\1\\1\\1\\1\\1\\1]),0,white",
// 		"3:MAX,2:0([1\\1\\1\\1\\1\\1\\1\\1]),1,white",
// 		"5:SUB,6:POW,0,white",
// 		"6:POW,7:MIN,0,white",
// 		"7:MIN,8:MIN,0,white",
// 		"8:MIN,3:MAX,0,white",
// 		"8:MIN,1:COS,1,white",
// 		"7:MIN,9:MIN,1,white",
// 		"9:MIN,10:ADD,0,white",
// 		"10:ADD,2:0([1\\1\\1\\1\\1\\1\\1\\1]),0,white",
// 		"10:ADD,0:MUL,1,white",
// 		"9:MIN,11:POW,1,white",
// 		"11:POW,12:SUB,0,white",
// 		"12:SUB,4:2([1\\1\\1\\1\\1\\1\\1\\1]),0,white",
// 		"12:SUB,2:0([1\\1\\1\\1\\1\\1\\1\\1]),1,white",
// 		"11:POW,2:0([1\\1\\1\\1\\1\\1\\1\\1]),1,white",
// 		"6:POW,13:DIV,1,white",
// 		"13:DIV,14:ADD,0,white",
// 		"14:ADD,15:ADD,0,white",
// 		"15:ADD,16:1([1\\1\\1\\1\\1\\1\\1\\1]),0,white",
// 		"15:ADD,2:0([1\\1\\1\\1\\1\\1\\1\\1]),1,white",
// 		"14:ADD,4:2([1\\1\\1\\1\\1\\1\\1\\1]),1,white",
// 		"13:DIV,17:SUB,1,white",
// 		"17:SUB,2:0([1\\1\\1\\1\\1\\1\\1\\1]),0,white",
// 		"17:SUB,16:1([1\\1\\1\\1\\1\\1\\1\\1]),1,white",
// 		"5:SUB,12:SUB,1,white",
// 		"18:MUL,19:MUL,0,white",
// 		"19:MUL,20:EQ,0,white",
// 		"20:EQ,21:DIV,0,white",
// 		"21:DIV,12:SUB,0,white",
// 		"21:DIV,0:MUL,1,white",
// 		"20:EQ,17:SUB,1,white",
// 		"19:MUL,1:COS,1,white",
// 		"18:MUL,22:MUL,1,white",
// 		"22:MUL,21:DIV,0,white",
// 		"22:MUL,14:ADD,1,white",
// 	};
// 	expectlines.sort();
// 	std::list<std::string> gotlines;
// 	std::string line;
// 	while (std::getline(ss, line))
// 	{
// 		gotlines.push_back(line);
// 	}
// 	gotlines.sort();
// 	std::vector<std::string> diffs;
// 	std::set_difference(expectlines.begin(), expectlines.end(),
// 		gotlines.begin(), gotlines.end(), std::back_inserter(diffs));

// 	std::stringstream diffstr;
// 	for (auto diff : diffs)
// 	{
// 		diffstr << diff << "\n";
// 	}
// 	std::string diffmsg = diffstr.str();
// 	EXPECT_EQ(0, diffmsg.size()) << "mismatching edges:\n" << diffmsg;
// }


#endif // DISABLE_OPTIMIZATION_TEST
