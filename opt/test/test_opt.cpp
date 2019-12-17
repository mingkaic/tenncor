
#ifndef DISABLE_OPTIMIZATION_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/functor.hpp"

#include "opt/mock/target.hpp"

#include "opt/parse.hpp"
#include "opt/filter.hpp"


static const std::string cst_rules =
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
"comm ADD(X,NEG(Y)) => SUB(X,Y);";


static const std::string edge_rules =
"REDUCE_SUM{special:[1,2]}(X) => X;"
"REDUCE_PROD{special2:[2,3]}(X) => X;"
"REDUCE_MIN{special3:[4,2]}(X) => X;"
"REDUCE_MAX{special4:[1,3]}(X) => X;"
"PERMUTE{spec:[0,1]}(X) => X;"
"EXTEND{spec:[1,0]}(X) => X;";


TEST(OPTIMIZE, PruneZeroSingles)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(std::vector<double>{}, shape, "special_var"));
	teq::TensptrT zero(new MockLeaf(std::vector<double>{}, shape, "0"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters empty;

	{
		auto wunfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var, zero},
			std::vector<double>{}, teq::Opcode{"POW", 0});
		auto zrofunc = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var},
			std::vector<double>{}, teq::Opcode{"POW", 0});
		auto opted = opt::optimize({wunfunc, zrofunc}, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect both optimized wunfunc to be 1
		EXPECT_GRAPHEQ("(constant:1[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);

		// expect both optimized zrofunc to be 0
		EXPECT_GRAPHEQ("(constant:0[2\\3\\4\\1\\1\\1\\1\\1])", opted[1]);
	}

	{
		auto lvfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var, zero},
			std::vector<double>{}, teq::Opcode{"ADD", 0});
		auto rvfunc = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var},
			std::vector<double>{}, teq::Opcode{"ADD", 0});
		auto opted = opt::optimize({lvfunc, rvfunc}, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect both optimized l and r vfuncs to be var
		std::string expect = "(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto lzero = std::make_shared<MockFunctor>(teq::TensptrsT{var, zero},
			std::vector<double>{}, teq::Opcode{"MUL", 0});
		auto rzero = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var},
			std::vector<double>{}, teq::Opcode{"MUL", 0});
		auto opted = opt::optimize({lzero, rzero}, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect both optimized l and r zeros to be 0
		std::string expect = "(constant:0[2\\3\\4\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto posvar = std::make_shared<MockFunctor>(teq::TensptrsT{var, zero},
			std::vector<double>{}, teq::Opcode{"SUB", 0});
		auto negvar = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var},
			std::vector<double>{}, teq::Opcode{"SUB", 0});
		auto opted = opt::optimize({posvar, negvar}, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect optimized posvar to be var
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opted[0]);

		// expect optimized negvar to be -var
		EXPECT_GRAPHEQ(
			"(NEG[2\\3\\4\\1\\1\\1\\1\\1])\n"
			" `--(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])", opted[1]);
	}

	{
		auto divz = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var},
			std::vector<double>{}, teq::Opcode{"DIV", 0});
		auto opted = opt::optimize({divz}, rules, empty);
		ASSERT_EQ(1, opted.size());

		// expect optimized divz to be zero
		EXPECT_GRAPHEQ("(constant:0[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);
	}

	{
		auto no_opt = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var},
			std::vector<double>{}, teq::Opcode{"MAX", 0});
		auto opted = opt::optimize({no_opt}, rules, empty);
		ASSERT_EQ(1, opted.size());

		// expect optimized not_opt to remain the same
		EXPECT_GRAPHEQ(
			"(MAX[2\\3\\4\\1\\1\\1\\1\\1])\n"
			" `--(constant:0[2\\3\\4\\1\\1\\1\\1\\1])\n"
			" `--(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);
	}
}


TEST(OPTIMIZE, PruneZeroGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(std::vector<double>{}, shape, "var"));
	teq::TensptrT var2(new MockLeaf(std::vector<double>{}, shape, "var2"));
	teq::TensptrT zero(new MockLeaf(std::vector<double>{}, shape, "0"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters empty;

	auto got1 = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var2},
		std::vector<double>{}, teq::Opcode{"ADD", 0});
	auto gotn1 = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var},
		std::vector<double>{}, teq::Opcode{"SUB", 0});
	auto got2 = std::make_shared<MockFunctor>(teq::TensptrsT{var2, zero},
		std::vector<double>{}, teq::Opcode{"SUB", 0});
	auto got22 = std::make_shared<MockFunctor>(teq::TensptrsT{var2, zero},
		std::vector<double>{}, teq::Opcode{"MAX", 0});

	auto too0 = std::make_shared<MockFunctor>(teq::TensptrsT{zero, got22},
		std::vector<double>{}, teq::Opcode{"MUL", 0});
	auto too = std::make_shared<MockFunctor>(teq::TensptrsT{var, too0},
		std::vector<double>{}, teq::Opcode{"ADD", 0});
	auto got11 = std::make_shared<MockFunctor>(teq::TensptrsT{got2, zero},
		std::vector<double>{}, teq::Opcode{"POW", 0});

	auto m0 = std::make_shared<MockFunctor>(teq::TensptrsT{got22, zero},
		std::vector<double>{}, teq::Opcode{"MAX", 0});
	auto m1 = std::make_shared<MockFunctor>(teq::TensptrsT{too, zero},
		std::vector<double>{}, teq::Opcode{"LT", 0});
	auto m = std::make_shared<MockFunctor>(teq::TensptrsT{m0, m1},
		std::vector<double>{}, teq::Opcode{"MIN", 0});

	auto nocascades0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{got1, gotn1}, std::vector<double>{}, teq::Opcode{"DIV", 0});
	auto nocascades1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{m, nocascades0}, std::vector<double>{}, teq::Opcode{"POW", 0});
	auto nocascades = std::make_shared<MockFunctor>(
		teq::TensptrsT{nocascades1, got2}, std::vector<double>{}, teq::Opcode{"SUB", 0});

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
}


TEST(OPTIMIZE, PropagateZeroGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(std::vector<double>{}, shape, "var"));
	teq::TensptrT var2(new MockLeaf(std::vector<double>{}, shape, "var2"));
	teq::TensptrT zero(new MockLeaf(std::vector<double>{}, shape, "0"));
	teq::TensptrT neg(new MockLeaf(std::vector<double>{}, shape, "-1"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters preconst;
	preconst.prenode_filters_.push_back(
		[&shape](teq::FuncptrT& f, opt::ParentReplF repl) -> teq::TensptrT
		{
			return opt::constant_func(f, repl,
				[&shape](teq::FuncptrT f) -> teq::TensptrT
				{
					if (f->get_opcode().name_ == "MAX")
					{
						return teq::TensptrT(new MockLeaf(std::vector<double>{}, shape, "0"));
					}
					return f;
				});
		});

	// max(pow(0, var2), -1) - var = 0
	// explanation:
	// pow(0, var2) reduces to 0 by optimization rule
	// max(0, -1) will not reduce by rule, but instead by preconst to 0
	// 0 - var reduces to -var by optimization rule
	auto negvar0 = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var2},
		std::vector<double>{}, teq::Opcode{"POW", 0});
	auto negvar1 = std::make_shared<MockFunctor>(teq::TensptrsT{neg, negvar0},
		std::vector<double>{}, teq::Opcode{"MAX", 0});
	auto negvar = std::make_shared<MockFunctor>(teq::TensptrsT{negvar1, var},
		std::vector<double>{}, teq::Opcode{"SUB", 0});

	auto opteds = opt::optimize({negvar}, rules, preconst);
	ASSERT_EQ(1, opteds.size());
	teq::TensptrT opted = opteds[0];
	EXPECT_GRAPHEQ(
		"(NEG[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])\n", opted);
}


TEST(OPTIMIZE, PruneOneSingles)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(std::vector<double>{}, shape, "special_var"));
	teq::TensptrT one(new MockLeaf(std::vector<double>{}, shape, "1"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters empty;

	{
		auto vfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var, one},
			std::vector<double>{}, teq::Opcode{"POW", 0});
		auto wunfunc = std::make_shared<MockFunctor>(teq::TensptrsT{one, var},
			std::vector<double>{}, teq::Opcode{"POW", 0});
		auto opted = opt::optimize({vfunc, wunfunc}, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect optimized vfunc to be 1
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opted[0]);

		// expect optimized wunfunc to be 1
		EXPECT_GRAPHEQ("(constant:1[2\\3\\4\\1\\1\\1\\1\\1])", opted[1]);
	}

	{
		auto lvfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var, one},
			std::vector<double>{}, teq::Opcode{"MUL", 0});
		auto rvfunc = std::make_shared<MockFunctor>(teq::TensptrsT{one, var},
			std::vector<double>{}, teq::Opcode{"MUL", 0});
		auto opted = opt::optimize({lvfunc, rvfunc}, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect both optimized l and r vfuncs to be var
		std::string expect = "(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto nomer = std::make_shared<MockFunctor>(teq::TensptrsT{var, one},
			std::vector<double>{}, teq::Opcode{"DIV", 0});
		auto opted = opt::optimize({nomer}, rules, empty);
		ASSERT_EQ(1, opted.size());

		// expect optimized nomer to be var
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	{
		auto wun = std::make_shared<MockFunctor>(teq::TensptrsT{var, var},
			std::vector<double>{}, teq::Opcode{"DIV", 0});
		auto opted = opt::optimize({wun}, rules, empty);
		ASSERT_EQ(1, opted.size());

		// expect optimized wun to be 1
		EXPECT_GRAPHEQ("(constant:1[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);
	}

	{
		auto no_opt = std::make_shared<MockFunctor>(teq::TensptrsT{one, var},
			std::vector<double>{}, teq::Opcode{"MAX", 0});
		auto opted = opt::optimize({no_opt}, rules, empty);
		ASSERT_EQ(1, opted.size());

		// expect optimized no_opt to remain the same
		EXPECT_GRAPHEQ(
			"(MAX[2\\3\\4\\1\\1\\1\\1\\1])\n"
			" `--(constant:1[2\\3\\4\\1\\1\\1\\1\\1])\n"
			" `--(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);
	}
}


TEST(OPTIMIZE, PruneOneGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(std::vector<double>{}, shape, "var"));
	teq::TensptrT one(new MockLeaf(std::vector<double>{}, shape, "1"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters empty;

	auto got1 = std::make_shared<MockFunctor>(teq::TensptrsT{one, var},
		std::vector<double>{}, teq::Opcode{"MUL", 0});
	auto got00 = std::make_shared<MockFunctor>(teq::TensptrsT{one, var},
		std::vector<double>{}, teq::Opcode{"POW", 0});
	auto got2 = std::make_shared<MockFunctor>(teq::TensptrsT{var, one},
		std::vector<double>{}, teq::Opcode{"MAX", 0});
	auto too = std::make_shared<MockFunctor>(teq::TensptrsT{one, got00},
		std::vector<double>{}, teq::Opcode{"ADD", 0});
	auto got11 = std::make_shared<MockFunctor>(teq::TensptrsT{var, one},
		std::vector<double>{}, teq::Opcode{"POW", 0});

	auto m0 = std::make_shared<MockFunctor>(teq::TensptrsT{one, too},
		std::vector<double>{}, teq::Opcode{"MAX", 0});
	auto m = std::make_shared<MockFunctor>(teq::TensptrsT{m0, got11},
		std::vector<double>{}, teq::Opcode{"MIN", 0});
	auto root0 = std::make_shared<MockFunctor>(teq::TensptrsT{got1, got2},
		std::vector<double>{}, teq::Opcode{"DIV", 0});
	auto root1 = std::make_shared<MockFunctor>(teq::TensptrsT{m, root0},
		std::vector<double>{}, teq::Opcode{"POW", 0});
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{root1, var},
		std::vector<double>{}, teq::Opcode{"SUB", 0});

	auto opteds = opt::optimize({root}, rules, empty);
	ASSERT_EQ(1, opteds.size());
	teq::TensptrT opted = opteds[0];
	EXPECT_GRAPHEQ(
		"(SUB[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(POW[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   `--(MAX[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(constant:1[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(ADD[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |       `--(constant:1[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |       `--(constant:1[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(DIV[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       `--(MAX[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |           `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |           `--(constant:1[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])", opted);
}


TEST(OPTIMIZE, PropagateOneGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(std::vector<double>{}, shape, "var"));
	teq::TensptrT var2(new MockLeaf(std::vector<double>{}, shape, "var2"));
	teq::TensptrT one(new MockLeaf(std::vector<double>{}, shape, "-1"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters preconst;
	preconst.prenode_filters_.push_back(
		[&shape](teq::FuncptrT& f, opt::ParentReplF repl) -> teq::TensptrT
		{
			return opt::constant_func(f, repl,
				[&shape](teq::FuncptrT f) -> teq::TensptrT
				{
					if (f->get_opcode().name_ == "SUB")
					{
						return teq::TensptrT(new MockLeaf(std::vector<double>{}, shape, "0"));
					}
					return f;
				});
		});

	// pow(var2, pow(var, div(var2, var2) - one)) = var2
	// explanation:
	// div(var2, var2) reduces to 1 by optimization rule
	// one - one will not reduce by rule, but instead by preconst to 0
	// pow(var, 0) var reduces to one by optimization rule
	// pow(var2, 1) var reduces to var2 by optimization rule
	auto posvar20 = std::make_shared<MockFunctor>(teq::TensptrsT{var2, var2},
		std::vector<double>{}, teq::Opcode{"DIV", 0});
	auto posvar21 = std::make_shared<MockFunctor>(teq::TensptrsT{posvar20, one},
		std::vector<double>{}, teq::Opcode{"SUB", 0});
	auto posvar22 = std::make_shared<MockFunctor>(teq::TensptrsT{var, posvar21},
		std::vector<double>{}, teq::Opcode{"POW", 0});
	auto posvar2 = std::make_shared<MockFunctor>(teq::TensptrsT{var2, posvar22},
		std::vector<double>{}, teq::Opcode{"POW", 0});

	auto opteds = opt::optimize({posvar2}, rules, preconst);
	ASSERT_EQ(1, opteds.size());
	teq::TensptrT opted = opteds[0];
	EXPECT_GRAPHEQ("(constant:var2[2\\3\\4\\1\\1\\1\\1\\1])\n", opted);
}


TEST(OPTIMIZE, PruneEdgeSingles)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(std::vector<double>{}, shape, "special_var"));
	teq::TensptrT var2(new MockLeaf(std::vector<double>{}, shape, "special_var2"));

	opt::CversionCtx rules = opt::parse(edge_rules, build_mock_target);
	opt::CustomFilters empty;

	{
		auto rule1 = std::make_shared<MockFunctor>(teq::TensptrsT{var},
			std::vector<double>{}, teq::Opcode{"REDUCE_SUM", 0});
		rule1->add_attr("special", std::make_unique<marsh::NumArray<double>>(
			std::vector<double>{1, 2}));
		auto opteds = opt::optimize({rule1}, rules, empty);
		ASSERT_EQ(1, opteds.size());
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opteds[0]);
	}

	// remove redundent reduced argument for non-empty shape
	{
		auto rule2 = std::make_shared<MockFunctor>(teq::TensptrsT{var},
			std::vector<double>{}, teq::Opcode{"REDUCE_PROD", 0});
		rule2->add_attr("special2", std::make_unique<marsh::NumArray<double>>(
			std::vector<double>{2, 3}));
		auto opteds = opt::optimize({rule2}, rules, empty);
		ASSERT_EQ(1, opteds.size());
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opteds[0]);
	}

	// don't reduce non-redundent reduced argument
	{
		auto not_rule2 = std::make_shared<MockFunctor>(teq::TensptrsT{var},
			std::vector<double>{}, teq::Opcode{"REDUCE_PROD", 0});
		not_rule2->add_attr("special2", std::make_unique<marsh::NumArray<double>>(
			std::vector<double>{2, 1}));
		auto opteds = opt::optimize({not_rule2}, rules, empty);
		ASSERT_EQ(1, opteds.size());
		EXPECT_GRAPHEQ(
			"(REDUCE_PROD[2\\3\\4\\1\\1\\1\\1\\1])\n"
			" `--(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opteds[0]);
	}
}


TEST(OPTIMIZE, PruneOpGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT zero(new MockLeaf(std::vector<double>{}, shape, "special_var0"));
	teq::TensptrT one(new MockLeaf(std::vector<double>{}, shape, "special_var"));
	teq::TensptrT two(new MockLeaf(std::vector<double>{}, shape, "special_var2"));
	teq::TensptrT three(new MockLeaf(std::vector<double>{}, shape, "special_var3"));

	opt::CversionCtx rules = opt::parse(edge_rules, build_mock_target);
	opt::CustomFilters empty;

	auto got1 = std::make_shared<MockFunctor>(teq::TensptrsT{three},
		std::vector<double>{}, teq::Opcode{"COS", 0});
	auto got30 = std::make_shared<MockFunctor>(teq::TensptrsT{one, three},
		std::vector<double>{}, teq::Opcode{"MUL", 0});
	auto got3 = std::make_shared<MockFunctor>(teq::TensptrsT{got30, two},
		std::vector<double>{}, teq::Opcode{"MUL", 0});
	auto gotn1 = std::make_shared<MockFunctor>(teq::TensptrsT{three, one},
		std::vector<double>{}, teq::Opcode{"SUB", 0});
	auto got2 = std::make_shared<MockFunctor>(teq::TensptrsT{two, three},
		std::vector<double>{}, teq::Opcode{"SUB", 0});
	auto got22 = std::make_shared<MockFunctor>(teq::TensptrsT{two, three},
		std::vector<double>{}, teq::Opcode{"MIN", 0});

	auto too0 = std::make_shared<MockFunctor>(teq::TensptrsT{zero},
		std::vector<double>{}, teq::Opcode{"REDUCE_PROD", 0});
	auto too1 = std::make_shared<MockFunctor>(teq::TensptrsT{too0},
		std::vector<double>{}, teq::Opcode{"REDUCE_PROD", 0});
	auto too2 = std::make_shared<MockFunctor>(teq::TensptrsT{got1, got22},
		std::vector<double>{}, teq::Opcode{"MUL", 0});
	auto too3 = std::make_shared<MockFunctor>(teq::TensptrsT{too2},
		std::vector<double>{}, teq::Opcode{"REDUCE_PROD", 0});
	auto too = std::make_shared<MockFunctor>(teq::TensptrsT{too2, too3},
		std::vector<double>{}, teq::Opcode{"MUL", 0});
	auto got11 = std::make_shared<MockFunctor>(teq::TensptrsT{got2, three},
		std::vector<double>{}, teq::Opcode{"POW", 0});

	auto m0 = std::make_shared<MockFunctor>(teq::TensptrsT{got22, got1},
		std::vector<double>{}, teq::Opcode{"MIN", 0});
	auto m1 = std::make_shared<MockFunctor>(teq::TensptrsT{too, got11},
		std::vector<double>{}, teq::Opcode{"MIN", 0});
	auto m = std::make_shared<MockFunctor>(teq::TensptrsT{m0, m1},
		std::vector<double>{}, teq::Opcode{"MIN", 0});

	auto to_opt0 = std::make_shared<MockFunctor>(teq::TensptrsT{got3, gotn1},
		std::vector<double>{}, teq::Opcode{"DIV", 0});
	auto to_opt1 = std::make_shared<MockFunctor>(teq::TensptrsT{m, to_opt0},
		std::vector<double>{}, teq::Opcode{"MIN", 0});
	auto to_opt = std::make_shared<MockFunctor>(teq::TensptrsT{to_opt1, got2},
		std::vector<double>{}, teq::Opcode{"SUB", 0});

	auto opteds = opt::optimize({to_opt}, rules, empty);
	ASSERT_EQ(1, opteds.size());
	auto root = opteds[0];

	EXPECT_GRAPHEQ(
		"(SUB[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   |   `--(constant:special_var2[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   |   `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |   `--(COS[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   |       `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |   `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       `--(MUL[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |   `--(MUL[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |   |   `--(COS[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |   |   |   `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |   |   `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |   |       `--(constant:special_var2[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |   |       `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |   `--(REDUCE_PROD[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |       `--(MUL[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |           `--(COS[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |           |   `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |           `--(MIN[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |               `--(constant:special_var2[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       |               `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |       `--(POW[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |           `--(SUB[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |           |   `--(constant:special_var2[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |           |   `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   |           `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |   `--(DIV[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       `--(MUL[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       |   `--(MUL[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       |   |   `--(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       |   |   `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       |   `--(constant:special_var2[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |       `--(SUB[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |           `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" |           `--(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(SUB[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"     `--(constant:special_var2[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"     `--(constant:special_var3[2\\3\\4\\1\\1\\1\\1\\1])",
		root);
}


TEST(OPTIMIZE, NearMatch)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(std::vector<double>{}, shape, "var"));
	teq::TensptrT one(new MockLeaf(std::vector<double>{}, shape, "1"));

	opt::CversionCtx rules = opt::parse(
		"DIV(1,comm ADD(1,NEG(EXP(NEG(X))))) => SIGMOID(X);", build_mock_target);
	opt::CustomFilters empty;

	// wrong rule, right equation, this should still fail to convert
	auto negvar = std::make_shared<MockFunctor>(teq::TensptrsT{var},
		std::vector<double>{}, teq::Opcode{"NEG", 0});
	auto envar = std::make_shared<MockFunctor>(teq::TensptrsT{negvar},
		std::vector<double>{}, teq::Opcode{"EXP", 0});
	auto oneadd = std::make_shared<MockFunctor>(teq::TensptrsT{one, envar},
		std::vector<double>{}, teq::Opcode{"ADD", 0});
	auto sigmoid = std::make_shared<MockFunctor>(teq::TensptrsT{one, oneadd},
		std::vector<double>{}, teq::Opcode{"DIV", 0});

	auto opteds = opt::optimize({sigmoid}, rules, empty);
	ASSERT_EQ(1, opteds.size());
	auto root = opteds[0];

	EXPECT_GRAPHEQ(
		"(DIV[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(constant:1[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(ADD[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"     `--(constant:1[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"     `--(EXP[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"         `--(NEG[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"             `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])",
		root);
}


#endif // DISABLE_OPTIMIZATION_TEST
