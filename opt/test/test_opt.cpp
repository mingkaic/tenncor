
#ifndef DISABLE_OPTIMIZATION_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "opt/mock/target.hpp"

#include "opt/parse.hpp"


TEST(OPTIMIZE, PruneZeroGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockTensor(shape, "var"));
	teq::TensptrT var2(new MockTensor(shape, "var2"));
	teq::TensptrT zero(new MockTensor(shape, "0"));

	opt::CversionCtx rules = opt::parse(
		"SUB(X,0) => X;"
		"SUB(0,X) => NEG(X);"
		"comm ADD(X,0) => X;"
		"comm MUL(X,0) => 0;"
		"comm MUL(X,1) => X;"
		"comm MUL(X,-1) => NEG(X);"
		"POW(0,X) => 0;"
		"POW(1,X) => 1;"
		"POW(X,0) => 1;"
		"POW(X,1) => X;"
		"DIV(0,X) => 0;"
		"DIV(X,1) => X;"
		"DIV(X,X) => 1;"
		"comm ADD(X,NEG(X)) => 0;"
		"comm ADD(X,NEG(Y)) => SUB(X,Y);", build_mock_target);

	auto got1 = std::make_shared<MockFunctor>(teq::Opcode{"ADD", 0}, teq::TensptrsT{zero, var2});
	auto gotn1 = std::make_shared<MockFunctor>(teq::Opcode{"SUB", 0}, teq::TensptrsT{zero, var});
	auto got2 = std::make_shared<MockFunctor>(teq::Opcode{"SUB", 0}, teq::TensptrsT{var2, zero});
	auto got22 = std::make_shared<MockFunctor>(teq::Opcode{"MAX", 0}, teq::TensptrsT{var2, zero});

	auto too0 = std::make_shared<MockFunctor>(teq::Opcode{"MUL", 0}, teq::TensptrsT{zero, got22});
	auto too = std::make_shared<MockFunctor>(teq::Opcode{"ADD", 0}, teq::TensptrsT{var, too0});
	auto got11 = std::make_shared<MockFunctor>(teq::Opcode{"POW", 0}, teq::TensptrsT{got2, zero});

	auto m0 = std::make_shared<MockFunctor>(teq::Opcode{"MAX", 0}, teq::TensptrsT{got22, zero});
	auto m1 = std::make_shared<MockFunctor>(teq::Opcode{"LT", 0}, teq::TensptrsT{too, zero});
	auto m = std::make_shared<MockFunctor>(teq::Opcode{"MIN", 0}, teq::TensptrsT{m0, m1});

	auto nocascades0 = std::make_shared<MockFunctor>(teq::Opcode{"DIV", 0}, teq::TensptrsT{got1, gotn1});
	auto nocascades1 = std::make_shared<MockFunctor>(teq::Opcode{"POW", 0}, teq::TensptrsT{m, nocascades0});
	auto nocascades = std::make_shared<MockFunctor>(teq::Opcode{"SUB", 0}, teq::TensptrsT{nocascades1, got2});

	opt::CustomFilters empty;
	auto opteds = opt::optimize({nocascades}, rules, empty);
	ASSERT_EQ(1, opteds.size());
	teq::TensptrT opted = opteds[0];
	EXPECT_GRAPHEQ(
		"(SUB[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(POW[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   `--(MAX[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(MAX[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   |   `--(constant:var2[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   |   `--(constant:0[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(constant:0[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   `--(LT[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       `--(constant:0[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(DIV[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       `--(constant:var2[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       `--(NEG[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |           `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(constant:var2[2\\3\\4\\1\\1\\1\\1\\1])\n", opted);

	// auto got0 = tenncor::tan(zero);
	// opted = opt::optimize({tenncor::pow(nocascades, got0)->get_tensor()}, rules);
	// ASSERT_EQ(1, opted.size());
	// EXPECT_GRAPHEQ("(constant:1[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);
}


// TEST(OPTIMIZE, PruneOneSingles)
// {
// 	eteq::NodeptrT<double> var = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(0, teq::Shape(),
// 		"special_var"));

// 	eteq::NodeptrT<double> one =
// 		eteq::make_constant_scalar<double>(1, teq::Shape());

// 	opt::CversionCtx rules = eteq::parse_file<double>("cfg/optimizations.rules");

// 	{
// 		auto vfunc = tenncor::pow(var, one);
// 		auto wunfunc = tenncor::pow(one, var);
// 		auto opted = opt::optimize({
// 			vfunc->get_tensor(),
// 			wunfunc->get_tensor(),
// 		}, rules);
// 		ASSERT_EQ(2, opted.size());
// 		// expect optimized vfunc to be 1
// 		EXPECT_GRAPHEQ("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
// 			opted[0]);

// 		// expect optimized wunfunc to be 1
// 		EXPECT_GRAPHEQ("(constant:1[1\\1\\1\\1\\1\\1\\1\\1])", opted[1]);
// 	}

// 	{
// 		auto lvfunc = var * one;
// 		auto rvfunc = one * var;
// 		auto opted = opt::optimize({
// 			lvfunc->get_tensor(),
// 			rvfunc->get_tensor(),
// 		}, rules);
// 		ASSERT_EQ(2, opted.size());
// 		// expect both optimized l and r vfuncs to be var
// 		std::string expect = "(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])";
// 		EXPECT_GRAPHEQ(expect, opted[0]);
// 		EXPECT_GRAPHEQ(expect, opted[1]);
// 	}

// 	{
// 		auto nomer = var / one;
// 		auto opted = opt::optimize({nomer->get_tensor()}, rules);
// 		ASSERT_EQ(1, opted.size());
// 		// expect optimized nomer to be var
// 		EXPECT_GRAPHEQ("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
// 			opted[0]);
// 	}

// 	{
// 		auto wun = var / var;
// 		auto opted = opt::optimize({wun->get_tensor()}, rules);
// 		ASSERT_EQ(1, opted.size());
// 		// expect optimized wun to be 1
// 		EXPECT_GRAPHEQ("(constant:1[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
// 	}

// 	{
// 		auto no_opt = tenncor::max(one, var);
// 		auto opted = opt::optimize({no_opt->get_tensor()}, rules);
// 		// expect optimized no_opt to remain the same
// 		EXPECT_GRAPHEQ(
// 			"(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 			" `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
// 	}
// }


// TEST(OPTIMIZE, PruneOneGraph)
// {
// 	eteq::NodeptrT<double> var = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(0, teq::Shape(),
// 		"var"));

// 	eteq::NodeptrT<double> one =
// 		eteq::make_constant_scalar<double>(1, teq::Shape());

// 	opt::CversionCtx rules = eteq::parse_file<double>("cfg/optimizations.rules");

// 	auto got0 = tenncor::log(one);
// 	auto got1 = tenncor::sqrt(one);
// 	auto got3 = one * var;
// 	auto got00 = tenncor::pow(one, var);
// 	auto got = tenncor::max(var, one);

// 	auto too = got1 + got0 * got00;
// 	auto got11 = tenncor::pow(var, one);

// 	auto m = tenncor::min(tenncor::max(got1, too), got11);
// 	auto root = tenncor::pow(m, got3 / got) - var;

// 	auto opted = opt::optimize({root->get_tensor()}, rules);
// 	ASSERT_EQ(1, opted.size());
// 	EXPECT_GRAPHEQ(
// 		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |       `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |       `--(MAX[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |           `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |           `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" `--(variable:var[1\\1\\1\\1\\1\\1\\1\\1])", opted[0]);
// }


// TEST(OPTIMIZE, PruneOpSingles)
// {
// 	eteq::NodeptrT<double> zero = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(0, teq::Shape(), "special_var0"));
// 	eteq::NodeptrT<double> one = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(1, teq::Shape({2, 3}), "special_var"));

// 	opt::CversionCtx rules = eteq::parse_file<double>("cfg/optimizations.rules");

// 	// merge redundent double reduced argument for empty shape
// 	{
// 		auto opted = opt::optimize({
// 			tenncor::reduce_sum(tenncor::reduce_sum(zero))->get_tensor(),
// 		}, rules);
// 		ASSERT_EQ(1, opted.size());
// 		EXPECT_GRAPHEQ("(variable:special_var0[1\\1\\1\\1\\1\\1\\1\\1])",
// 			opted[0]);
// 	}

// 	// merge redundent double reduced argument for non-empty shape
// 	{
// 		auto opted = opt::optimize({
// 			tenncor::reduce_sum(tenncor::reduce_sum(one))->get_tensor(),
// 		}, rules);
// 		ASSERT_EQ(1, opted.size());
// 		EXPECT_GRAPHEQ(
// 			"(REDUCE_SUM[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 			" `--(variable:special_var[2\\3\\1\\1\\1\\1\\1\\1])",
// 			opted[0]);
// 	}

// 	// don't merge non-redundent double reduced argument
// 	{
// 		auto opted = opt::optimize({
// 			tenncor::reduce_sum(tenncor::reduce_sum(one, 1), 0)->get_tensor(),
// 		}, rules);
// 		ASSERT_EQ(1, opted.size());
// 		EXPECT_GRAPHEQ(
// 			"(REDUCE_SUM[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 			" `--(REDUCE_SUM[2\\1\\1\\1\\1\\1\\1\\1])\n"
// 			"     `--(variable:special_var[2\\3\\1\\1\\1\\1\\1\\1])",
// 			opted[0]);
// 	}

// 	// don't merge mul-reduced_add
// 	{
// 		auto opted = opt::optimize({
// 			(tenncor::reduce_sum(one) * zero)->get_tensor(),
// 		}, rules);
// 		ASSERT_EQ(1, opted.size());
// 		EXPECT_GRAPHEQ(
// 			"(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 			" `--(REDUCE_SUM[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 			" |   `--(variable:special_var[2\\3\\1\\1\\1\\1\\1\\1])\n"
// 			" `--(variable:special_var0[1\\1\\1\\1\\1\\1\\1\\1])",
// 			opted[0]);
// 	}
// }


// TEST(OPTIMIZE, PruneOpGraph)
// {
// 	eteq::NodeptrT<double> zero = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(0, teq::Shape({3, 4}), "special_var0"));
// 	eteq::NodeptrT<double> one = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(1, teq::Shape(), "special_var"));
// 	eteq::NodeptrT<double> two = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(2, teq::Shape(), "special_var2"));
// 	eteq::NodeptrT<double> three = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(3, teq::Shape(), "special_var3"));

// 	auto got1 = tenncor::cos(three);
// 	auto got3 = (one * three) * two;
// 	auto gotn1 = three - one;
// 	auto got2 = two - three;
// 	auto got22 = tenncor::min(two, three);

// 	auto too =
// 		tenncor::reduce_prod(tenncor::reduce_prod(zero, 0), 1) *
// 		tenncor::reduce_prod(got1 * got22);
// 	auto got11 = tenncor::pow(got2, three);

// 	auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));

// 	opt::CversionCtx rules = eteq::parse_file<double>("cfg/optimizations.rules");

// 	auto opted = opt::optimize({
// 		(tenncor::min(m, got3 / gotn1) - got2)->get_tensor(),
// 	}, rules);
// 	ASSERT_EQ(1, opted.size());
// 	auto root = opted[0];

// 	EXPECT_GRAPHEQ(
// 		"(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   |   |   `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   |   |   `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   |   `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   |       `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |   `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       |   `--(REDUCE_PROD[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       |   |   `--(variable:special_var0[3\\4\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       |   `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       |       `--(COS[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       |       |   `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       |       `--(MIN[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       |           `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       |           `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |       `--(POW[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |           `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |           |   `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |           |   `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   |           `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |   `--(DIV[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |       `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |       |   `--(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |       |   |   `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |       |   |   `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |       |   `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |       `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |           `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" |           `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		" `--(SUB[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		"     `--(variable:special_var2[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 		"     `--(variable:special_var3[1\\1\\1\\1\\1\\1\\1\\1])",
// 		opted[0]);
// }


// TEST(OPTIMIZE, GroupSingles)
// {
// 	eteq::NodeptrT<double> one = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(1, teq::Shape(), "special_var"));
// 	eteq::NodeptrT<double> two = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(2, teq::Shape(), "special_var2"));

// 	opt::CversionCtx rules = eteq::parse_file<double>("cfg/optimizations.rules");

// 	// mul and div and next to each level
// 	{
// 		auto opted = opt::optimize({((one / two) * two)->get_tensor()}, rules);
// 		ASSERT_EQ(1, opted.size());
// 		EXPECT_GRAPHEQ("(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
// 			opted[0]);
// 	}

// 	// mul and div are separated by a level
// 	{
// 		auto opted = opt::optimize({
// 			(((one / two) * one) * two)->get_tensor(),
// 		}, rules);
// 		ASSERT_EQ(1, opted.size());
// 		EXPECT_GRAPHEQ(
// 			"(MUL[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])\n"
// 			" `--(variable:special_var[1\\1\\1\\1\\1\\1\\1\\1])",
// 			opted[0]);
// 	}
// }


// TEST(OPTIMIZE, ReuseOpGraph)
// {
// 	eteq::NodeptrT<double> zero = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(0, teq::Shape()));
// 	eteq::NodeptrT<double> one = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(1, teq::Shape()));
// 	eteq::NodeptrT<double> two = eteq::convert_to_node(
// 		eteq::make_variable_scalar<double>(2, teq::Shape()));

// 	eteq::NodeptrT<double> root;
// 	{
// 		auto got1 = tenncor::cos(zero);
// 		auto got3 = (one + zero) + two;
// 		auto gotn1 = zero - one;
// 		auto got2 = two - zero;
// 		auto got22 = tenncor::max(two, zero);

// 		auto too = zero + got1 * got22;
// 		auto got11 = tenncor::pow(got2, zero);

// 		auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));
// 		root = tenncor::pow(m, got3 / gotn1) - got2;
// 	}

// 	eteq::NodeptrT<double> subroot;
// 	{
// 		auto other_got1 = tenncor::cos(zero);
// 		auto got22 = tenncor::max(two, zero);
// 		subroot = other_got1 * got22;
// 	}

// 	eteq::NodeptrT<double> copyroot;
// 	{
// 		auto got1 = tenncor::cos(zero);
// 		auto got3 = (one + zero) + two;
// 		auto gotn1 = zero - one;
// 		auto got2 = two - zero;
// 		auto got22 = tenncor::max(two, zero);

// 		auto too = zero + got1 * got22;
// 		auto got11 = tenncor::pow(got2, zero);

// 		auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));
// 		copyroot = tenncor::pow(m, got3 / gotn1) - got2;
// 	}

// 	eteq::NodeptrT<double> splitroot;
// 	{
// 		auto got1 = tenncor::cos(zero);
// 		auto got3 = (one + zero) + two;
// 		auto gotn1 = zero - one;
// 		auto got2 = two - zero;
// 		auto got22 = tenncor::max(two, zero);

// 		auto too = got2 / (got1 * got22);
// 		auto got11 = too == gotn1;

// 		splitroot = (got11 * got1) * (too * got3);
// 	}

// 	opt::CversionCtx empty_rules = eteq::parse<double>("");

// 	auto opted = opt::optimize({
// 		subroot->get_tensor(),
// 		root->get_tensor(),
// 		splitroot->get_tensor(),
// 		copyroot->get_tensor(),
// 	}, empty_rules);
// 	auto opt_subroot = opted[0];
// 	auto opt_root = opted[1];
// 	auto opt_splitroot = opted[2];
// 	auto opt_copyroot = opted[3];

// 	ASSERT_NE(nullptr, opt_subroot);
// 	ASSERT_NE(nullptr, opt_root);
// 	ASSERT_NE(nullptr, opt_splitroot);
// 	ASSERT_NE(nullptr, opt_copyroot);

// 	std::stringstream ss;
// 	CSVEquation ceq;
// 	opt_subroot->accept(ceq);
// 	opt_root->accept(ceq);
// 	opt_splitroot->accept(ceq);
// 	opt_copyroot->accept(ceq);
// 	ceq.to_stream(ss);

// 	std::list<std::string> expectlines =
// 	{
// 		"0:MUL,1:COS,0,white",
// 		"0:MUL,3:MAX,1,white",
// 		"10:ADD,0:MUL,1,white",
// 		"10:ADD,2:0,0,white",
// 		"11:POW,12:SUB,0,white",
// 		"11:POW,2:0,1,white",
// 		"12:SUB,2:0,1,white",
// 		"12:SUB,4:2,0,white",
// 		"13:DIV,14:ADD,0,white",
// 		"13:DIV,17:SUB,1,white",
// 		"14:ADD,15:ADD,0,white",
// 		"14:ADD,4:2,1,white",
// 		"15:ADD,16:1,0,white",
// 		"15:ADD,2:0,1,white",
// 		"17:SUB,16:1,1,white",
// 		"17:SUB,2:0,0,white",
// 		"18:MUL,19:MUL,0,white",
// 		"18:MUL,22:MUL,1,white",
// 		"19:MUL,1:COS,1,white",
// 		"19:MUL,20:EQ,0,white",
// 		"1:COS,2:0,0,white",
// 		"20:EQ,17:SUB,1,white",
// 		"20:EQ,21:DIV,0,white",
// 		"21:DIV,0:MUL,1,white",
// 		"21:DIV,12:SUB,0,white",
// 		"22:MUL,14:ADD,1,white",
// 		"22:MUL,21:DIV,0,white",
// 		"3:MAX,2:0,1,white",
// 		"3:MAX,4:2,0,white",
// 		"5:SUB,12:SUB,1,white",
// 		"5:SUB,6:POW,0,white",
// 		"6:POW,13:DIV,1,white",
// 		"6:POW,7:MIN,0,white",
// 		"7:MIN,8:MIN,0,white",
// 		"7:MIN,9:MIN,1,white",
// 		"8:MIN,1:COS,1,white",
// 		"8:MIN,3:MAX,0,white",
// 		"9:MIN,10:ADD,0,white",
// 		"9:MIN,11:POW,1,white",
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
// 	std::set_symmetric_difference(expectlines.begin(), expectlines.end(),
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
