
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
	teq::TensptrT var(new MockLeaf(shape, "special_var"));
	teq::TensptrT zero(new MockLeaf(shape, "0"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters empty;

	{
		auto wunfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var, zero}, teq::Opcode{"POW", 0});
		auto zrofunc = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var}, teq::Opcode{"POW", 0});
		teq::TensptrsT opted = {wunfunc, zrofunc};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect both optimized wunfunc to be 1
		EXPECT_GRAPHEQ("(constant:1[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);

		// expect both optimized zrofunc to be 0
		EXPECT_GRAPHEQ("(constant:0[2\\3\\4\\1\\1\\1\\1\\1])", opted[1]);
	}

	{
		auto lvfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var, zero}, teq::Opcode{"ADD", 0});
		auto rvfunc = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var}, teq::Opcode{"ADD", 0});
		teq::TensptrsT opted = {lvfunc, rvfunc};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect both optimized l and r vfuncs to be var
		std::string expect = "(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto lzero = std::make_shared<MockFunctor>(teq::TensptrsT{var, zero}, teq::Opcode{"MUL", 0});
		auto rzero = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var}, teq::Opcode{"MUL", 0});
		teq::TensptrsT opted = {lzero, rzero};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect both optimized l and r zeros to be 0
		std::string expect = "(constant:0[2\\3\\4\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto posvar = std::make_shared<MockFunctor>(teq::TensptrsT{var, zero}, teq::Opcode{"SUB", 0});
		auto negvar = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var}, teq::Opcode{"SUB", 0});
		teq::TensptrsT opted = {posvar, negvar};
		opt::optimize(opted, rules, empty);
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
		auto divz = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var}, teq::Opcode{"DIV", 0});
		teq::TensptrsT opted = {divz};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(1, opted.size());

		// expect optimized divz to be zero
		EXPECT_GRAPHEQ("(constant:0[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);
	}

	{
		auto no_opt = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var}, teq::Opcode{"MAX", 0});
		teq::TensptrsT opted = {no_opt};
		opt::optimize(opted, rules, empty);
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
	teq::TensptrT var(new MockLeaf(shape, "var"));
	teq::TensptrT var2(new MockLeaf(shape, "var2"));
	teq::TensptrT zero(new MockLeaf(shape, "0"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters empty;

	auto got1 = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var2}, teq::Opcode{"ADD", 0});
	auto gotn1 = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var}, teq::Opcode{"SUB", 0});
	auto got2 = std::make_shared<MockFunctor>(teq::TensptrsT{var2, zero}, teq::Opcode{"SUB", 0});
	auto got22 = std::make_shared<MockFunctor>(teq::TensptrsT{var2, zero}, teq::Opcode{"MAX", 0});

	auto too0 = std::make_shared<MockFunctor>(teq::TensptrsT{zero, got22}, teq::Opcode{"MUL", 0});
	auto too = std::make_shared<MockFunctor>(teq::TensptrsT{var, too0}, teq::Opcode{"ADD", 0});
	auto got11 = std::make_shared<MockFunctor>(teq::TensptrsT{got2, zero}, teq::Opcode{"POW", 0});

	auto m0 = std::make_shared<MockFunctor>(teq::TensptrsT{got22, zero}, teq::Opcode{"MAX", 0});
	auto m1 = std::make_shared<MockFunctor>(teq::TensptrsT{too, zero}, teq::Opcode{"LT", 0});
	auto m = std::make_shared<MockFunctor>(teq::TensptrsT{m0, m1}, teq::Opcode{"MIN", 0});

	auto nocascades0 = std::make_shared<MockFunctor>(
		teq::TensptrsT{got1, gotn1}, teq::Opcode{"DIV", 0});
	auto nocascades1 = std::make_shared<MockFunctor>(
		teq::TensptrsT{m, nocascades0}, teq::Opcode{"POW", 0});
	auto nocascades = std::make_shared<MockFunctor>(
		teq::TensptrsT{nocascades1, got2}, teq::Opcode{"SUB", 0});

	teq::TensptrsT opted = {nocascades};
	opt::optimize(opted, rules, empty);
	ASSERT_EQ(1, opted.size());
	teq::TensptrT o = opted[0];
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
		" `--(constant:var2[2\\3\\4\\1\\1\\1\\1\\1])\n", o);
}


TEST(OPTIMIZE, PropagateZeroGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(shape, "var"));
	teq::TensptrT var2(new MockLeaf(shape, "var2"));
	teq::TensptrT zero(new MockLeaf(shape, "0"));
	teq::TensptrT neg(new MockLeaf(shape, "-1"));

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
						return teq::TensptrT(new MockLeaf(shape, "0"));
					}
					return f;
				});
		});

	// max(pow(0, var2), -1) - var = 0
	// explanation:
	// pow(0, var2) reduces to 0 by optimization rule
	// max(0, -1) will not reduce by rule, but instead by preconst to 0
	// 0 - var reduces to -var by optimization rule
	auto negvar0 = std::make_shared<MockFunctor>(teq::TensptrsT{zero, var2}, teq::Opcode{"POW", 0});
	auto negvar1 = std::make_shared<MockFunctor>(teq::TensptrsT{neg, negvar0}, teq::Opcode{"MAX", 0});
	auto negvar = std::make_shared<MockFunctor>(teq::TensptrsT{negvar1, var}, teq::Opcode{"SUB", 0});

	teq::TensptrsT opted = {negvar};
	opt::optimize(opted, rules, preconst);
	ASSERT_EQ(1, opted.size());
	teq::TensptrT o = opted[0];
	EXPECT_GRAPHEQ(
		"(NEG[2\\3\\4\\1\\1\\1\\1\\1])\n"
		" `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])\n", o);
}


TEST(OPTIMIZE, PruneOneSingles)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(shape, "special_var"));
	teq::TensptrT one(new MockLeaf(shape, "1"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters empty;

	{
		auto vfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var, one}, teq::Opcode{"POW", 0});
		auto wunfunc = std::make_shared<MockFunctor>(teq::TensptrsT{one, var}, teq::Opcode{"POW", 0});
		teq::TensptrsT opted = {vfunc, wunfunc};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect optimized vfunc to be 1
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opted[0]);

		// expect optimized wunfunc to be 1
		EXPECT_GRAPHEQ("(constant:1[2\\3\\4\\1\\1\\1\\1\\1])", opted[1]);
	}

	{
		auto lvfunc = std::make_shared<MockFunctor>(teq::TensptrsT{var, one}, teq::Opcode{"MUL", 0});
		auto rvfunc = std::make_shared<MockFunctor>(teq::TensptrsT{one, var}, teq::Opcode{"MUL", 0});
		teq::TensptrsT opted = {lvfunc, rvfunc};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(2, opted.size());

		// expect both optimized l and r vfuncs to be var
		std::string expect = "(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])";
		EXPECT_GRAPHEQ(expect, opted[0]);
		EXPECT_GRAPHEQ(expect, opted[1]);
	}

	{
		auto nomer = std::make_shared<MockFunctor>(teq::TensptrsT{var, one}, teq::Opcode{"DIV", 0});
		teq::TensptrsT opted = {nomer};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(1, opted.size());

		// expect optimized nomer to be var
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	{
		auto wun = std::make_shared<MockFunctor>(teq::TensptrsT{var, var}, teq::Opcode{"DIV", 0});
		teq::TensptrsT opted = {wun};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(1, opted.size());

		// expect optimized wun to be 1
		EXPECT_GRAPHEQ("(constant:1[2\\3\\4\\1\\1\\1\\1\\1])", opted[0]);
	}

	{
		auto no_opt = std::make_shared<MockFunctor>(teq::TensptrsT{one, var}, teq::Opcode{"MAX", 0});
		teq::TensptrsT opted = {no_opt};
		opt::optimize(opted, rules, empty);
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
	teq::TensptrT var(new MockLeaf(shape, "var"));
	teq::TensptrT one(new MockLeaf(shape, "1"));

	opt::CversionCtx rules = opt::parse(cst_rules, build_mock_target);
	opt::CustomFilters empty;

	auto got1 = std::make_shared<MockFunctor>(teq::TensptrsT{one, var}, teq::Opcode{"MUL", 0});
	auto got00 = std::make_shared<MockFunctor>(teq::TensptrsT{one, var}, teq::Opcode{"POW", 0});
	auto got2 = std::make_shared<MockFunctor>(teq::TensptrsT{var, one}, teq::Opcode{"MAX", 0});
	auto too = std::make_shared<MockFunctor>(teq::TensptrsT{one, got00}, teq::Opcode{"ADD", 0});
	auto got11 = std::make_shared<MockFunctor>(teq::TensptrsT{var, one}, teq::Opcode{"POW", 0});

	auto m0 = std::make_shared<MockFunctor>(teq::TensptrsT{one, too}, teq::Opcode{"MAX", 0});
	auto m = std::make_shared<MockFunctor>(teq::TensptrsT{m0, got11}, teq::Opcode{"MIN", 0});
	auto root0 = std::make_shared<MockFunctor>(teq::TensptrsT{got1, got2}, teq::Opcode{"DIV", 0});
	auto root1 = std::make_shared<MockFunctor>(teq::TensptrsT{m, root0}, teq::Opcode{"POW", 0});
	auto root = std::make_shared<MockFunctor>(teq::TensptrsT{root1, var}, teq::Opcode{"SUB", 0});

	teq::TensptrsT opted = {root};
	opt::optimize(opted, rules, empty);
	ASSERT_EQ(1, opted.size());
	teq::TensptrT o = opted[0];
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
		" `--(constant:var[2\\3\\4\\1\\1\\1\\1\\1])", o);
}


TEST(OPTIMIZE, PropagateOneGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(shape, "var"));
	teq::TensptrT var2(new MockLeaf(shape, "var2"));
	teq::TensptrT one(new MockLeaf(shape, "-1"));

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
						return teq::TensptrT(new MockLeaf(shape, "0"));
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
	auto posvar20 = std::make_shared<MockFunctor>(teq::TensptrsT{var2, var2}, teq::Opcode{"DIV", 0});
	auto posvar21 = std::make_shared<MockFunctor>(teq::TensptrsT{posvar20, one}, teq::Opcode{"SUB", 0});
	auto posvar22 = std::make_shared<MockFunctor>(teq::TensptrsT{var, posvar21}, teq::Opcode{"POW", 0});
	auto posvar2 = std::make_shared<MockFunctor>(teq::TensptrsT{var2, posvar22}, teq::Opcode{"POW", 0});

	teq::TensptrsT opted = {posvar2};
	opt::optimize(opted, rules, preconst);
	ASSERT_EQ(1, opted.size());
	teq::TensptrT o = opted[0];
	EXPECT_GRAPHEQ("(constant:var2[2\\3\\4\\1\\1\\1\\1\\1])\n", o);
}


TEST(OPTIMIZE, PruneEdgeSingles)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT var(new MockLeaf(shape, "special_var"));
	teq::TensptrT var2(new MockLeaf(shape, "special_var2"));

	opt::CversionCtx rules = opt::parse(edge_rules, build_mock_target);
	opt::CustomFilters empty;

	{
		auto rule1 = std::make_shared<MockFunctor>(teq::TensptrsT{var}, teq::Opcode{"REDUCE_SUM", 0});
		rule1->add_attr("special", std::make_unique<marsh::NumArray<double>>(
			std::vector<double>{1, 2}));
		teq::TensptrsT opted = {rule1};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	// remove redundent reduced argument for non-empty shape
	{
		auto rule2 = std::make_shared<MockFunctor>(teq::TensptrsT{var}, teq::Opcode{"REDUCE_PROD", 0});
		rule2->add_attr("special2", std::make_unique<marsh::NumArray<double>>(
			std::vector<double>{2, 3}));
		teq::TensptrsT opted = {rule2};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ("(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opted[0]);
	}

	// don't reduce non-redundent reduced argument
	{
		auto not_rule2 = std::make_shared<MockFunctor>(teq::TensptrsT{var}, teq::Opcode{"REDUCE_PROD", 0});
		not_rule2->add_attr("special2", std::make_unique<marsh::NumArray<double>>(
			std::vector<double>{2, 1}));
		teq::TensptrsT opted = {not_rule2};
		opt::optimize(opted, rules, empty);
		ASSERT_EQ(1, opted.size());
		EXPECT_GRAPHEQ(
			"(REDUCE_PROD[2\\3\\4\\1\\1\\1\\1\\1])\n"
			" `--(constant:special_var[2\\3\\4\\1\\1\\1\\1\\1])",
			opted[0]);
	}
}


TEST(OPTIMIZE, PruneOpGraph)
{
	teq::Shape shape({2, 3, 4});
	teq::TensptrT zero(new MockLeaf(shape, "special_var0"));
	teq::TensptrT one(new MockLeaf(shape, "special_var"));
	teq::TensptrT two(new MockLeaf(shape, "special_var2"));
	teq::TensptrT three(new MockLeaf(shape, "special_var3"));

	opt::CversionCtx rules = opt::parse(edge_rules, build_mock_target);
	opt::CustomFilters empty;

	auto got1 = std::make_shared<MockFunctor>(teq::TensptrsT{three}, teq::Opcode{"COS", 0});
	auto got30 = std::make_shared<MockFunctor>(teq::TensptrsT{one, three}, teq::Opcode{"MUL", 0});
	auto got3 = std::make_shared<MockFunctor>(teq::TensptrsT{got30, two}, teq::Opcode{"MUL", 0});
	auto gotn1 = std::make_shared<MockFunctor>(teq::TensptrsT{three, one}, teq::Opcode{"SUB", 0});
	auto got2 = std::make_shared<MockFunctor>(teq::TensptrsT{two, three}, teq::Opcode{"SUB", 0});
	auto got22 = std::make_shared<MockFunctor>(teq::TensptrsT{two, three}, teq::Opcode{"MIN", 0});

	auto too0 = std::make_shared<MockFunctor>(teq::TensptrsT{zero}, teq::Opcode{"REDUCE_PROD", 0});
	auto too1 = std::make_shared<MockFunctor>(teq::TensptrsT{too0}, teq::Opcode{"REDUCE_PROD", 0});
	auto too2 = std::make_shared<MockFunctor>(teq::TensptrsT{got1, got22}, teq::Opcode{"MUL", 0});
	auto too3 = std::make_shared<MockFunctor>(teq::TensptrsT{too2}, teq::Opcode{"REDUCE_PROD", 0});
	auto too = std::make_shared<MockFunctor>(teq::TensptrsT{too2, too3}, teq::Opcode{"MUL", 0});
	auto got11 = std::make_shared<MockFunctor>(teq::TensptrsT{got2, three}, teq::Opcode{"POW", 0});

	auto m0 = std::make_shared<MockFunctor>(teq::TensptrsT{got22, got1}, teq::Opcode{"MIN", 0});
	auto m1 = std::make_shared<MockFunctor>(teq::TensptrsT{too, got11}, teq::Opcode{"MIN", 0});
	auto m = std::make_shared<MockFunctor>(teq::TensptrsT{m0, m1}, teq::Opcode{"MIN", 0});

	auto to_opt0 = std::make_shared<MockFunctor>(teq::TensptrsT{got3, gotn1}, teq::Opcode{"DIV", 0});
	auto to_opt1 = std::make_shared<MockFunctor>(teq::TensptrsT{m, to_opt0}, teq::Opcode{"MIN", 0});
	auto to_opt = std::make_shared<MockFunctor>(teq::TensptrsT{to_opt1, got2}, teq::Opcode{"SUB", 0});

	teq::TensptrsT opted = {to_opt};
	opt::optimize(opted, rules, empty);
	ASSERT_EQ(1, opted.size());
	auto root = opted[0];

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
	teq::TensptrT var(new MockLeaf(shape, "var"));
	teq::TensptrT one(new MockLeaf(shape, "1"));

	opt::CversionCtx rules = opt::parse(
		"DIV(1,comm ADD(1,NEG(EXP(NEG(X))))) => SIGMOID(X);", build_mock_target);
	opt::CustomFilters empty;

	// wrong rule, right equation, this should still fail to convert
	auto negvar = std::make_shared<MockFunctor>(teq::TensptrsT{var}, teq::Opcode{"NEG", 0});
	auto envar = std::make_shared<MockFunctor>(teq::TensptrsT{negvar}, teq::Opcode{"EXP", 0});
	auto oneadd = std::make_shared<MockFunctor>(teq::TensptrsT{one, envar}, teq::Opcode{"ADD", 0});
	auto sigmoid = std::make_shared<MockFunctor>(teq::TensptrsT{one, oneadd}, teq::Opcode{"DIV", 0});

	teq::TensptrsT opted = {sigmoid};
	opt::optimize(opted, rules, empty);
	ASSERT_EQ(1, opted.size());
	auto root = opted[0];

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
