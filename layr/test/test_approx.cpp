
#ifndef DISABLE_APPROX_TEST


#include "gtest/gtest.h"

#include "dbg/print/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "eteq/make.hpp"

#include "query/query.hpp"

#include "layr/approx.hpp"


TEST(APPROX, StochasticGD)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto root = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "root");

	auto groups = layr::sgd<PybindT>(layr::VarMapT<PybindT>{{
		eteq::VarptrT<PybindT>(leaf),
		eteq::ETensor<PybindT>(root)}}, 0.67);
	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(constant:0.67[1\\1\\1\\1\\1\\1\\1\\1])",
		groups.begin()->second);
}


TEST(APPROX, RmsMomentum)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto root = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "root");

	auto groups = layr::rms_momentum<PybindT>(layr::VarMapT<PybindT>{{
		eteq::VarptrT<PybindT>(leaf),
		eteq::ETensor<PybindT>(root)}}, 0.67, 0.78,
		std::numeric_limits<PybindT>::epsilon());
	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |       `--(constant:0.67[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"     `--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(SQRT[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |   `--(ASSIGN[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |       `--(variable:momentum[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |       `--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |           `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |           |   `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |           |   |   `--(constant:0.78[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |           |   `--(variable:momentum[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |           `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |               `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |               |   `--(constant:0.22[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |               `--(SQUARE[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |                   `--(variable:root[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"             `--(constant:1.19209e-07[1\\1\\1\\1\\1\\1\\1\\1])",
		groups.begin()->second);
}


TEST(APPROX, GroupAssign)
{
	teq::Shape shape({5});

	auto leaf = eteq::make_variable_scalar<PybindT>(0, shape, "leaf");
	auto err = eteq::make_variable_scalar<PybindT>(0.5, shape, "err");

	layr::VarMapT<PybindT> groups = layr::rms_momentum<PybindT>(
		layr::VarMapT<PybindT>{{leaf, eteq::ETensor<PybindT>(err)}},
		1, 0.52, std::numeric_limits<PybindT>::epsilon());

	auto sess = eigen::get_session();
	sess.track(teq::TensptrsT{groups.begin()->second});
	sess.update();

	// momentm is calculated first:
	// init momentum = 1
	// momentum = 0.52 * momentum + 0.48 * err^2
	// = 0.52 + 0.48 * 0.25 = 0.64
	// leaf is calculated next:
	// leaf - 1 * err / sqrt(momentum)
	// = 0 - 1 * 0.5 / sqrt(0.64) = -0.5 / 0.8 = -0.625

	// otherwise if momentum is calculated after
	// leaf - 1 * err / sqrt(momentum)
	// = 0 - 1 * 0.5 / sqrt(1) = -0.5
	std::stringstream ss;
	ss << "{\"leaf\":{\"label\":\"momentum\"}}";
	query::QResultsT results;
	query::search::OpTrieT itable;
	query::search::populate_itable(itable, {groups.begin()->second});
	query::Query(itable).where(ss).exec(results);

	PybindT* o = (PybindT*) static_cast<eteq::Variable<PybindT>*>(
		results.front().root_)->device().data();
	std::vector<PybindT> expecto(shape.n_elems(), 0.64);
	std::vector<PybindT> ovec(o, o + shape.n_elems());
	EXPECT_VECEQ(expecto, ovec);

	PybindT* d = (PybindT*) leaf->device().data();
	PybindT exdval = -0.625;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exdval - d[i]) / exdval));
	}
}


#endif // DISABLE_APPROX_TEST
