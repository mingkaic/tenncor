
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
	auto err = tenncor<PybindT>().abs(leaf);

	auto groups = tenncor<PybindT>().approx.sgd(
		err, eteq::EVariablesT<PybindT>{leaf}, 0.67);
	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___`--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(ABS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___|_______`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___`--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________`--(constant:0.67[1\\1\\1\\1\\1\\1\\1\\1])",
		groups.begin()->second);
}


TEST(APPROX, Adagrad)
{
	std::vector<teq::DimT> slist = {18, 9, 3};

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, teq::Shape(slist), "leaf");
	auto err = tenncor<PybindT>().abs(leaf);

	auto groups = tenncor<PybindT>().approx.adagrad(
		err, eteq::EVariablesT<PybindT>{leaf}, 0.67);
	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_`--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___|___|___`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___|___|___`--(ABS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___|___|_______`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|___`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(constant:0.67[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________`--(SQRT[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|___`--(ASSIGN_ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|_______`--(variable:momentum[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|_______`--(SQUARE[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|___________`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|_______________`--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|_______________|___`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|_______________|___`--(ABS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|_______________|_______`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________|_______________`--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________`--(constant:1.19209e-07[1\\1\\1\\1\\1\\1\\1\\1])",
		groups.begin()->second);
}


TEST(APPROX, Adadelta)
{
	std::vector<teq::DimT> slist = {18, 9, 3};
	teq::Shape shape(slist);

	auto leaf = eteq::make_variable_scalar<PybindT>(
		0, shape, "leaf");
	auto err = tenncor<PybindT>().sin(leaf);

	PybindT step_rate = 1;
	PybindT decay = 0.91;
	PybindT offset = 0.16;
	auto groups = tenncor<PybindT>().approx.adadelta(
		err, eteq::EVariablesT<PybindT>{leaf}, step_rate, decay, offset);
	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|___|___`--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___|___`--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______`--(SQRT[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______|___`--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______|_______`--(variable:ex_sqr_delx[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______|_______`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______|___________`--(constant:0.16[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___|_______`--(SQRT[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|___________`--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________`--(ASSIGN[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|___`--(variable:ex_sqr_grad[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|___`--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|_______`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|_______|___`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|_______|___|___`--(constant:0.91[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|_______|___`--(variable:ex_sqr_grad[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|_______`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|___________`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|___________|___`--(constant:0.09[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|___________`--(SQUARE[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|_______________`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|___________________`--(COS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|___________________|___`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________|___________________`--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|_______________`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|___|___________________`--(constant:0.16[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_|___`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|_______`--(COS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|_______|___`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_|_______`--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_`--(DEPENDENCIES[1\\1\\1\\1\\1\\1\\1\\1])"
		"_____`--(ASSIGN[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________`--(variable:ex_sqr_delx[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________`--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________|___`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________|___|___`--(constant:0.91[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________|___`--(variable:ex_sqr_delx[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________|___`--(constant:0.09[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________________`--(SQUARE[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________________`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|___`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|___|___`--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________________________|___`--(DIV[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______`--(SQRT[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______|___`--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______|_______`--(variable:ex_sqr_delx[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______|_______`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______|___________`--(constant:0.16[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________________________|_______`--(SQRT[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|___________`--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________`--(ASSIGN[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|___`--(variable:ex_sqr_grad[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|___`--(ADD[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|_______`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|_______|___`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|_______|___|___`--(constant:0.91[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|_______|___`--(variable:ex_sqr_grad[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|_______`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|___________`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|___________|___`--(constant:0.09[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|___________`--(SQUARE[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|_______________`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|___________________`--(COS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|___________________|___`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________|___________________`--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|_______________`--(EXTEND[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_________________________|___________________`--(constant:0.16[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________________________`--(MUL[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________________________`--(COS[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________________________|___`--(variable:leaf[18\\9\\3\\1\\1\\1\\1\\1])\n"
		"_____________________________`--(constant:1[18\\9\\3\\1\\1\\1\\1\\1])",
		groups.begin()->second);

	// evaluating execution order
	auto sess = eigen::get_session();
	sess.track(teq::TensptrSetT{groups.begin()->second});
	sess.update();

	eteq::Variable<PybindT>* g;
	{
		std::stringstream ss;
		ss << "{\"leaf\":{\"label\":\"ex_sqr_grad\"}}";
		query::Query itable;
		groups.begin()->second->accept(itable);
		query::Node cond;
		query::json_parse(cond, ss);
		auto results = itable.match(cond);
		g = static_cast<eteq::Variable<PybindT>*>(results.front().root_);
	}

	eteq::Variable<PybindT>* d;
	{
		std::stringstream ss;
		ss << "{\"leaf\":{\"label\":\"ex_sqr_delx\"}}";
		query::Query itable;
		groups.begin()->second->accept(itable);
		query::Node cond;
		query::json_parse(cond, ss);
		auto results = itable.match(cond);
		d = static_cast<eteq::Variable<PybindT>*>(results.front().root_);
	}

	// g = decay * g + (1 - decay) * f'(x) ^ 2
	// dx = step_rate * sqrt(d + offset) / sqrt(g + offset) * f'(x)
	// d = decay * d + (1 - decay) * dx ^ 2
	// x_next = x - dx

	// init g = 0, d = 0, decay = 0.91, offset = 0.16
	// f'(x) = cos(0) = 1

	// g is calculated first:
	// g = 0.91 * g + 0.09 * f'(x) ^ 2
	//   = 0.91 * 1 = 0.09

	// d and x are calculated next using same common value c:
	// c = 1 * sqrt(d + 0.16) / sqrt(g + 0.16) * f'(x)
	//   = sqrt(0.16) / sqrt(0.09 + 0.16)
	//   = 0.8
	// d = 0.91 * d + 0.09 * c ^ 2
	//   = 0.09 * 0.8 ^ 2
	//   = 0.09 * 0.64
	//   = 0.0576
	// x = x - c
	//   = 0 - 0.8
	//   = -0.8

	// otherwise if g is calculated after
	// d = 0.91 * d + 0.09 * (1 * sqrt(d + 0.16) / sqrt(g + 0.16) * f'(x)) ^ 2
	//   = 0.09 * (sqrt(0.16) / sqrt(0.16)) ^ 2
	//   = 0.09
	// x = x - 1 * sqrt(d + 0.16) / sqrt(g + 0.16) * f'(x)
	//   = 0 - sqrt(0.16) / sqrt(0.16)
	//   = -1

	// OR if d is calculated before x (by re-evaluating c)
	// d = 0.0288
	// x = x - 1 * sqrt(d + 0.16) / sqrt(g + 0.16) * f'(x)
	//   = 0 - sqrt(0.1888) / sqrt(0.09 + 0.16)
	//   = -0.43451121964  / sqrt(0.25)
	//   = -0.86902243929

	PybindT* og = (PybindT*) g->device().data();
	PybindT exog = 0.09;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exog - og[i]) / exog)) <<
			"expect: " << exog << ", got: " << og[i];
	}

	PybindT* od = (PybindT*) d->device().data();
	PybindT exod = 0.0576;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exod - od[i]) / exod)) <<
			"expect: " << exod << ", got: " << od[i];
	}

	PybindT* data = (PybindT*) leaf->device().data();
	PybindT exdval = -0.8;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exdval - data[i]) / exdval)) <<
			"expect: " << exdval << ", got: " << data[i];
	}
}


TEST(APPROX, RmsMomentum)
{
	teq::Shape shape({5});

	auto leaf = eteq::make_variable_scalar<PybindT>(0, shape, "leaf");
	auto err = tenncor<PybindT>().sin(leaf) / 2.f;

	PybindT learning_rate = 1.;
	PybindT discount_rate = 0.52;
	layr::VarErrsT<PybindT> groups = tenncor<PybindT>().approx.rms_momentum(
		err, eteq::EVariablesT<PybindT>{leaf},
		learning_rate, discount_rate,
		std::numeric_limits<PybindT>::epsilon());
	ASSERT_EQ(1, groups.size());
	EXPECT_GRAPHEQ(
		"(ASSIGN_SUB[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_`--(variable:leaf[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_`--(DIV[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(COS[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___|___`--(variable:leaf[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___`--(DIV[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|_______`--(constant:1[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|_______`--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___|___________`--(constant:2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|___`--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____|_______`--(constant:1[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____`--(ADD[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(SQRT[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___`--(ASSIGN[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______`--(variable:momentum[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______`--(ADD[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___________`--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___________|___`--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___________|___|___`--(constant:0.52[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___________|___`--(variable:momentum[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___________`--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______________`--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______________|___`--(constant:0.48[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______________`--(SQUARE[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___________________`--(MUL[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______________________`--(COS[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______________________|___`--(variable:leaf[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______________________`--(DIV[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___________________________`--(constant:1[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|___________________________`--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________|_______________________________`--(constant:2[1\\1\\1\\1\\1\\1\\1\\1])\n"
		"_________`--(EXTEND[5\\1\\1\\1\\1\\1\\1\\1])\n"
		"_____________`--(constant:1.19209e-07[1\\1\\1\\1\\1\\1\\1\\1])",
		groups[0].second);

	// evaluating execution order
	auto sess = eigen::get_session();
	sess.track(teq::TensptrSetT{groups[0].second});
	sess.update();

	eteq::Variable<PybindT>* momentum;
	{
		std::stringstream ss;
		ss << "{\"leaf\":{\"label\":\"momentum\"}}";
		query::Query itable;
		groups[0].second->accept(itable);
		query::Node cond;
		query::json_parse(cond, ss);
		auto results = itable.match(cond);
		momentum = static_cast<eteq::Variable<PybindT>*>(
			results.front().root_);
	}

	// init momentum = 1

	// momentm is calculated first:
	// momentum = 0.52 * momentum + 0.48 * err^2
	//          = 0.52 + 0.48 * 0.25 = 0.64

	// leaf is calculated next:
	// leaf = leaf - 1 * err / sqrt(momentum)
	//      = 0 - 1 * 0.5 / sqrt(0.64) = -0.5 / 0.8 = -0.625

	// otherwise if momentum is calculated after
	// leaf = leaf - 1 * err / sqrt(momentum)
	//      = 0 - 1 * 0.5 / sqrt(1) = -0.5

	PybindT* o = (PybindT*) momentum->device().data();
	std::vector<PybindT> expecto(shape.n_elems(), 0.64);
	std::vector<PybindT> ovec(o, o + shape.n_elems());
	EXPECT_VECEQ(expecto, ovec);

	PybindT* d = (PybindT*) leaf->device().data();
	PybindT exdval = -0.625;
	for (size_t i = 0, n = shape.n_elems(); i < n; ++i)
	{
		ASSERT_GT(0.001, abs((exdval - d[i]) / exdval)) <<
			"expect: " << exdval << ", got: " << d[i];
	}
}


#endif // DISABLE_APPROX_TEST
