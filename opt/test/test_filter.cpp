#define DISABLE_FILTER_TEST
#ifndef DISABLE_FILTER_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/opfunc.hpp"

#include "opt/filter.hpp"


TEST(FILTER, CalcConstants)
{
	eteq::NodeptrT<double> var = eteq::convert_to_node(
		eteq::make_variable_scalar<double>(0, teq::Shape(),
		"special_var"));

	eteq::NodeptrT<double> two =
		eteq::make_constant_scalar<double>(2, teq::Shape());
	eteq::NodeptrT<double> three =
		eteq::make_constant_scalar<double>(3, teq::Shape());
	eteq::NodeptrT<double> four =
		eteq::make_constant_scalar<double>(4, teq::Shape());

	opt::CversionCtx empty_rules = eteq::constant_funcs<double>("");

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
		auto adv_func = (tenncor::sin(var) + tenncor::sin(two)) *
			tenncor::pow(three, four);
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


TEST(FILTER, ReuseOpGraph)
{
	eteq::NodeptrT<double> zero = eteq::convert_to_node(
		eteq::make_variable_scalar<double>(0, teq::Shape()));
	eteq::NodeptrT<double> one = eteq::convert_to_node(
		eteq::make_variable_scalar<double>(1, teq::Shape()));
	eteq::NodeptrT<double> two = eteq::convert_to_node(
		eteq::make_variable_scalar<double>(2, teq::Shape()));

	eteq::NodeptrT<double> root;
	{
		auto got1 = tenncor::cos(zero);
		auto got3 = (one + zero) + two;
		auto gotn1 = zero - one;
		auto got2 = two - zero;
		auto got22 = tenncor::max(two, zero);

		auto too = zero + got1 * got22;
		auto got11 = tenncor::pow(got2, zero);

		auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));
		root = tenncor::pow(m, got3 / gotn1) - got2;
	}

	eteq::NodeptrT<double> subroot;
	{
		auto other_got1 = tenncor::cos(zero);
		auto got22 = tenncor::max(two, zero);
		subroot = other_got1 * got22;
	}

	eteq::NodeptrT<double> copyroot;
	{
		auto got1 = tenncor::cos(zero);
		auto got3 = (one + zero) + two;
		auto gotn1 = zero - one;
		auto got2 = two - zero;
		auto got22 = tenncor::max(two, zero);

		auto too = zero + got1 * got22;
		auto got11 = tenncor::pow(got2, zero);

		auto m = tenncor::min(tenncor::min(got22, got1), tenncor::min(too, got11));
		copyroot = tenncor::pow(m, got3 / gotn1) - got2;
	}

	eteq::NodeptrT<double> splitroot;
	{
		auto got1 = tenncor::cos(zero);
		auto got3 = (one + zero) + two;
		auto gotn1 = zero - one;
		auto got2 = two - zero;
		auto got22 = tenncor::max(two, zero);

		auto too = got2 / (got1 * got22);
		auto got11 = too == gotn1;

		splitroot = (got11 * got1) * (too * got3);
	}

	opt::CversionCtx empty_rules = eteq::parse<double>("");

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


#endif // DISABLE_FILTER_TEST
