
#ifndef DISABLE_APPROX_TEST


#include "gtest/gtest.h"

#include "dbg/stream/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "eteq/make.hpp"

#include "layr/approx.hpp"


TEST(APPROX, StochasticGD)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto root = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "root");

	auto groups = layr::sgd<PybindT>(layr::VarErrsT<PybindT>{{leaf,
		eteq::ETensor<PybindT>(root)}}, 0.67);
	ASSERT_EQ(1, groups.size());

	auto ass = groups.at(0);
	ASSERT_EQ(1, ass.size());

	auto assign = ass.at(0);
	EXPECT_EQ(assign.target_.get(), leaf.get());
	EXPECT_GRAPHEQ(
		"(SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(constant:0.67[1\\1\\1\\1\\1\\1\\1\\1])",
		assign.source_);
}


TEST(APPROX, RmsMomentum)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto root = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "root");

	auto groups = layr::rms_momentum<PybindT>(layr::VarErrsT<PybindT>{{leaf,
		eteq::ETensor<PybindT>(root)}}, 0.67, 0.78,
		std::numeric_limits<PybindT>::epsilon());
	ASSERT_EQ(2, groups.size());

	auto mom_ass = groups.at(0);
	ASSERT_EQ(1, mom_ass.size());

	auto var_ass = groups.at(1);
	ASSERT_EQ(1, var_ass.size());

	auto mom_assign = mom_ass.at(0);
	auto mom = mom_assign.target_.get();
	EXPECT_NE(mom, leaf.get());
	EXPECT_STREQ("momentum", mom->to_string().c_str());
	EXPECT_GRAPHEQ(
		"(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" |   `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" |   |   `--(constant:0.78[1\\1\\1\\1\\1\\1\\1\\1])\n"
		" |   `--(variable:momentum[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(constant:0.22[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"     `--(SQUARE[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])",
		mom_assign.source_);

	auto var_assign = var_ass.at(0);
	EXPECT_EQ(var_assign.target_.get(), leaf.get());
	EXPECT_GRAPHEQ(
		"(SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |       `--(constant:0.67[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"     `--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(SQRT[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |   `--(variable:momentum[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"             `--(constant:1.19209e-07[1\\1\\1\\1\\1\\1\\1\\1])",
		var_assign.source_);
}


TEST(APPROX, GroupAssign)
{
	teq::Shape shape({5});

	auto leaf = eteq::make_variable_scalar<PybindT>(0, shape, "leaf");
	auto err = eteq::make_variable_scalar<PybindT>(1, shape, "err");

	auto groups = layr::rms_momentum<PybindT>(layr::VarErrsT<PybindT>{{leaf,
		eteq::ETensor<PybindT>(err)}}, 0.67, 0.78,
		std::numeric_limits<PybindT>::epsilon());

	teq::Session sess;
	teq::TensptrsT track_batch;
	for (layr::AssignsT<PybindT>& assigns : groups)
	{
		for (layr::VarAssign<PybindT>& assign : assigns)
		{
			track_batch.push_back(assign.source_);
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
			eteq::VarptrT<PybindT>(err)->assign(arr);
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
	PybindT* t = (PybindT*) err->data();
	std::vector<PybindT> expectt(shape.n_elems(), 2);
	std::vector<PybindT> tvec(t, t + shape.n_elems());
	EXPECT_VECEQ(expectt, tvec);

	PybindT* o = (PybindT*) static_cast<eteq::Variable<PybindT>*>(
		updated_order[0])->data();
	std::vector<PybindT> expecto(shape.n_elems(), 1);
	std::vector<PybindT> ovec(o, o + shape.n_elems());
	EXPECT_VECEQ(expecto, ovec);

	PybindT* d = (PybindT*) leaf->data();
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

	auto groups = layr::rms_momentum<PybindT>(layr::VarErrsT<PybindT>{{leaf,
		eteq::ETensor<PybindT>(err)}}, 0.67, 0.78,
		std::numeric_limits<PybindT>::epsilon());

	teq::Session sess;
	teq::TensptrsT track_batch;
	for (layr::AssignsT<PybindT>& assigns : groups)
	{
		for (layr::VarAssign<PybindT>& assign : assigns)
		{
			track_batch.push_back(assign.source_);
		}
	}
	sess.track(track_batch);
	layr::assign_groups_preupdate(groups,
		[&](teq::TensSetT& to_update)
		{
			teq::ShapedArr<PybindT> arr(shape, 2);
			eteq::VarptrT<PybindT>(err)->assign(arr);
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
	PybindT* t = (PybindT*) err->data();
	std::vector<PybindT> expectt(shape.n_elems(), 2);
	std::vector<PybindT> tvec(t, t + shape.n_elems());
	EXPECT_VECEQ(expectt, tvec);

	ASSERT_EQ(2, groups.size());
	ASSERT_EQ(1, groups[0].size());
	PybindT* o = (PybindT*) groups[0][0].target_->data();
	PybindT exoval = 1.66;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exoval - o[i]) / exoval));
	}

	PybindT* d = (PybindT*) leaf->data();
	PybindT exdval = -1.04004170445;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exdval - d[i]) / exdval));
	}
}


#endif // DISABLE_APPROX_TEST
