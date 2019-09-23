
#ifndef DISABLE_APPROX_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "exam/exam.hpp"

#include "eteq/variable.hpp"

#include "layr/err_approx.hpp"


TEST(APPROX, StochasticGD)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto root = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "root");

	auto groups = layr::sgd(layr::VarErrsT{{leaf,
		eteq::convert_to_node(root)}}, 0.67, "stuff");
	ASSERT_EQ(1, groups.size());

	auto ass = groups.at(0);
	ASSERT_EQ(1, ass.size());

	auto assign = ass.at(0);
	EXPECT_STREQ("sgd::stuff_grad_leaf", assign.label_.c_str());
	EXPECT_EQ(assign.target_->get_tensor().get(), leaf->get_tensor().get());
	EXPECT_GRAPHEQ(
		"(SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(constant:0.67[18\\9\\3\\1\\1\\1\\1\\1])",
		assign.source_->get_tensor());
}


TEST(APPROX, RMS_Momentum)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto root = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "root");

	auto groups = layr::rms_momentum(layr::VarErrsT{{leaf,
		eteq::convert_to_node(root)}}, 0.67, 0.78,
		std::numeric_limits<PybindT>::epsilon(), "stuff");
	ASSERT_EQ(2, groups.size());

	auto mom_ass = groups.at(0);
	ASSERT_EQ(1, mom_ass.size());

	auto var_ass = groups.at(1);
	ASSERT_EQ(1, var_ass.size());

	auto mom_assign = mom_ass.at(0);
	EXPECT_STREQ("rms_momentum::stuff_momentum_leaf", mom_assign.label_.c_str());
	auto mom = mom_assign.target_->get_tensor().get();
	EXPECT_NE(mom, leaf->get_tensor().get());
	EXPECT_STREQ("momentum", mom->to_string().c_str());
	EXPECT_GRAPHEQ(
		"(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:momentum[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" |   `--(constant:0.78[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(SQUARE[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(constant:0.22[18\\9\\3\\1\\1\\1\\1\\1])",
		mom_assign.source_->get_tensor());

	auto var_assign = var_ass.at(0);
	EXPECT_STREQ("rms_momentum::stuff_grad_leaf", var_assign.label_.c_str());
	EXPECT_EQ(var_assign.target_->get_tensor().get(), leaf->get_tensor().get());
	EXPECT_GRAPHEQ(
		"(SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(constant:0.67[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(SQRT[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |   `--(variable:momentum[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(constant:1.19209e-07[18\\9\\3\\1\\1\\1\\1\\1])",
		var_assign.source_->get_tensor());
}


#endif // DISABLE_APPROX_TEST
