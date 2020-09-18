
#ifndef DISABLE_HOSVC_OPTIMIZE_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "tenncor/hone/hosvc/hosvc.hpp"


TEST(OPTIMIZE, Cstrules)
{
	// tensor operation
	teq::DimsT slist = {2, 3, 4};
	std::vector<double> data = {
		59, 10, 28, 10, 67, 62, 23, 4, 55, 77, 28, 16,
		82, 52, 47, 16, 7, 85, 37, 2, 8, 52, 62, 43
	};
	std::vector<double> data2 = {
		22, 15, 74, 38, 61, 95, 62, 81, 99, 76, 7, 22,
		56, 50, 19, 13, 12, 10, 31, 40, 60, 54, 6, 83
	};
	teq::Shape shape(slist);
	teq::NElemT n = shape.n_elems();
	assert(data.size() == n);
	assert(data2.size() == n);

	teq::TensptrT a = eteq::make_variable<double>(data.data(), shape, "a");
	teq::TensptrT b = eteq::make_constant<double>(data2.data(), shape);
	teq::TensptrT c = eteq::make_constant_scalar<double>(4, shape);

	// instance 1
	distr::iDistrMgrptrT mgr(make_mgr(5112, "mgr1"));

	teq::TensptrT lhs(eteq::make_functor(egen::ADD, {b, c}));

	auto& svc = distr::get_iosvc(*mgr);
	std::string id = svc.expose_node(lhs);

	// instance 2
	distr::iDistrMgrptrT mgr2 = make_mgr(5113, "mgr2");
	auto& svc2 = distr::get_iosvc(*mgr2);

	error::ErrptrT err = nullptr;
	teq::TensptrT lhs_ref = svc2.lookup_node(err, id);
	ASSERT_NOERR(err);
	teq::TensptrT rhs(eteq::make_functor(egen::ADD, {a, b}));
	teq::TensptrT everyone(eteq::make_functor(egen::ADD, {lhs_ref, rhs}));

	opt::Optimization optimization;
	auto optres = distr::get_hosvc(*mgr2).(optimize, {everyone});
	ASSERT_EQ(1, optres.size());
	everyone = optres.front();

	auto refs = distr::reachable_refs(teq::TensptrsT{everyone});
	ASSERT_EQ(1, refs.size());
	auto lhs_repl = refs.front();

	EXPECT_GRAPHEQ(fmts::sprintf(
		"(ADD<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(placeholder:%s<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n"
		"_`--(constant:[81\\25\\102\\48\\128\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n",
		lhs_repl->to_string().c_str()), everyone);

	auto lhs_repl_id = lhs_repl->node_id();
	auto opt_lhs = svc.lookup_node(err, lhs_repl_id, false);
	ASSERT_NOERR(err);
	ASSERT_NE(nullptr, opt_lhs);

	EXPECT_GRAPHEQ(
		"(constant:[26\\19\\78\\42\\65\\...]<DOUBLE>[2\\3\\4\\1\\1\\1\\1\\1])\n", opt_lhs);
}


#endif // DISABLE_HOSVC_OPTIMIZE_TEST
