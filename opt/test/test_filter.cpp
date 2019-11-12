#define DISABLE_FILTER_TEST
#ifndef DISABLE_FILTER_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "teq/mock/leaf.hpp"
#include "teq/mock/opfunc.hpp"

#include "opt/filter.hpp"


TEST(OPTIMIZE, CalcConstants)
{
	// teq::Shape shape({2, 3, 4});
	// teq::TensptrT a(new MockTensor(shape));
	// auto f = std::make_shared<MockOpfunc>(a, teq::Opcode{"SIN", egen::SIN});

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


#endif // DISABLE_FILTER_TEST
