
#ifndef DISABLE_APPROX_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

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


TEST(APPROX, RmsMomentum)
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
		" |   `--(constant:0.78[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:momentum[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(constant:0.22[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(SQUARE[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])",
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


TEST(APPROX, GroupAssign)
{
	teq::Shape shape({5});

	auto leaf = eteq::make_variable_scalar<PybindT>(0, shape, "leaf");
	auto err = eteq::make_variable_scalar<PybindT>(1, shape, "err");

	auto groups = layr::rms_momentum(layr::VarErrsT{{leaf,
		eteq::convert_to_node(err)}}, 0.67, 0.78,
		std::numeric_limits<PybindT>::epsilon(), "stuff");

	teq::Session sess;
	teq::TensptrsT track_batch;
	for (layr::AssignsT& assigns : groups)
	{
		for (layr::VarAssign& assign : assigns)
		{
			track_batch.push_back(assign.source_->get_tensor());
		}
	}
	sess.track(track_batch);
	sess.update();

	teq::TensT updated_order;
	updated_order.reserve(2);
	layr::assign_groups(groups,
		[&](teq::TensSetT& updated)
		{
			ASSERT_EQ(1, updated.size());
			updated_order.push_back(*updated.begin());
			teq::ShapedArr<PybindT> arr(shape, 2);
			err->assign(arr);
			sess.update(teq::TensSetT(updated_order.begin(),
				updated_order.end()));
		});
	ASSERT_EQ(2, updated_order.size());
	EXPECT_EQ("momentum", updated_order[0]->to_string());
	EXPECT_EQ("leaf", updated_order[1]->to_string());

	// err = 1, leaf = 0
	// momentm is calculated first:
	// momentum = 0.78 * 1 + 0.22 * err = 1
	// err is updated next:
	// err = 2
	// leaf is calculated next:
	// leaf - 0.67 * err / sqrt(momentum) =
	// 0 - 0.67 * 2 / sqrt(1) = -1.34
	PybindT* t = err->data();
	std::vector<PybindT> expectt(shape.n_elems(), 2);
	std::vector<PybindT> tvec(t, t + shape.n_elems());
	EXPECT_VECEQ(expectt, tvec);

	PybindT* o = (PybindT*) static_cast<eteq::Variable<PybindT>*>(
		updated_order[0])->data();
	std::vector<PybindT> expecto(shape.n_elems(), 1);
	std::vector<PybindT> ovec(o, o + shape.n_elems());
	EXPECT_VECEQ(expecto, ovec);

	PybindT* d = leaf->data();
	PybindT exdval = -1.34;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exdval - d[i]) / exdval));
	}
}


TEST(APPROX, PreUpdateGroupAssign)
{
	teq::Shape shape({5});

	auto leaf = eteq::make_variable_scalar<PybindT>(0, shape, "leaf");
	auto err = eteq::make_variable_scalar<PybindT>(1, shape, "err");

	auto groups = layr::rms_momentum(layr::VarErrsT{{leaf,
		eteq::convert_to_node(err)}}, 0.67, 0.78,
		std::numeric_limits<PybindT>::epsilon(), "stuff");

	teq::Session sess;
	teq::TensptrsT track_batch;
	for (layr::AssignsT& assigns : groups)
	{
		for (layr::VarAssign& assign : assigns)
		{
			track_batch.push_back(assign.source_->get_tensor());
		}
	}
	sess.track(track_batch);
	layr::assign_groups_preupdate(groups,
		[&](teq::TensSetT& to_update)
		{
			teq::ShapedArr<PybindT> arr(shape, 2);
			err->assign(arr);
			ASSERT_EQ(1, to_update.size());
			sess.update_target(to_update);
		});

	// err = 1, leaf = 0
	// err is updated first:
	// err = 2
	// momentm is calculated next:
	// momentum = 0.78 * 1 + 0.22 * sqrt(err) =
	// 0.78 * 1 + 0.22 * 4 = 1.66
	// err is updated again (redundant):
	// err = 2
	// leaf is calculated next:
	// leaf - 0.67 * err / sqrt(momentum) =
	// 0 - 0.67 * 2 / sqrt(1.66) = (ugly #)
	PybindT* t = err->data();
	std::vector<PybindT> expectt(shape.n_elems(), 2);
	std::vector<PybindT> tvec(t, t + shape.n_elems());
	EXPECT_VECEQ(expectt, tvec);

	ASSERT_EQ(2, groups.size());
	ASSERT_EQ(1, groups[0].size());
	PybindT* o = groups[0][0].target_->data();
	PybindT exoval = 1.66;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exoval - o[i]) / exoval));
	}

	PybindT* d = leaf->data();
	PybindT exdval = -1.04004170445;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exdval - d[i]) / exdval));
	}
}


#endif // DISABLE_APPROX_TEST
