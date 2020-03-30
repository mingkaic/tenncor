
#ifndef DISABLE_APPROX_TEST


#include "gtest/gtest.h"

#include "dbg/print/teq_csv.hpp"

#include "testutil/tutil.hpp"

#include "eteq/make.hpp"

#include "query/query.hpp"
#include "query/parse.hpp"

#include "layr/approx.hpp"

#include "generated/api.hpp"
#include "generated/pyapi.hpp"


TEST(APPROX, StochasticGD)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto err = tenncor::abs(leaf);

	auto groups = tenncor::approx::sgd<PybindT>(
		err, eteq::EVariablesT<PybindT>{leaf}, 0.67);
	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   |   `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   |   `--(ABS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   |       `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(constant:0.67[1\\1\\1\\1\\1\\1\\1\\1])",
		groups.begin()->second);
}


TEST(APPROX, RmsMomentum)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto err = tenncor::abs(leaf);

	auto groups = tenncor::approx::rms_momentum<PybindT>(
		err, eteq::EVariablesT<PybindT>{leaf}, 0.67, 0.78,
		std::numeric_limits<PybindT>::epsilon());
	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		" `--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   |   `--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   |   |   `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   |   |   `--(ABS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   |   |       `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"     |   |   `--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
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
		"         |                   `--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |                       `--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |                       |   `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |                       |   `--(ABS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |                       |       `--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         |                       `--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"         `--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"             `--(constant:1.19209e-07[1\\1\\1\\1\\1\\1\\1\\1])",
		groups.begin()->second);
}


TEST(APPROX, GroupAssign)
{
	teq::Shape shape({5});

	auto leaf = eteq::make_variable_scalar<PybindT>(0, shape, "leaf");
	auto err = tenncor::sin(leaf) / 2.f;

	layr::VarMapT<PybindT> groups = tenncor::approx::rms_momentum<PybindT>(
		err, eteq::EVariablesT<PybindT>{leaf},
		1, 0.52, std::numeric_limits<PybindT>::epsilon());

	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[5\\1\\1\\1\\1\\1\\1\\1])\n"
		" `--(variable:leaf[5\\1\\1\\1\\1\\1\\1\\1])\n"
		" `--(DIV[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     `--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |   `--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |   |   `--(COS[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |   |   |   `--(variable:leaf[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |   |   `--(DIV[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |   |       `--(constant:1[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |   |       `--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |   |           `--(constant:2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |   `--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"     |       `--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"     `--(ADD[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         `--(SQRT[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |   `--(ASSIGN[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |       `--(variable:momentum[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |       `--(ADD[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |           `--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |           |   `--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |           |   |   `--(constant:0.52[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |           |   `--(variable:momentum[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |           `--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |               `--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |               |   `--(constant:0.48[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |               `--(SQUARE[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                   `--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                       `--(COS[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                       |   `--(variable:leaf[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                       `--(DIV[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                           `--(constant:1[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                           `--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"         |                               `--(constant:2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"         `--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"             `--(constant:1.19209e-07[1\\1\\1\\1\\1\\1\\1\\1])\n",
		groups.begin()->second);

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
	query::Query itable;
	groups.begin()->second->accept(itable);
	query::Node cond;
	query::json_parse(cond, ss);
	auto results = itable.match(cond);

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
