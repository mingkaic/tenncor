
#ifndef DISABLE_HOSVC_OPTIMIZE_TEST


#include "gtest/gtest.h"

#include "testutil/tutil.hpp"

#include "dbg/print/printsvc/printsvc.hpp"

#include "tenncor/distr/mock/mock.hpp"

#include "tenncor/hone/hosvc/hosvc.hpp"


const std::string test_service = "tenncor.hone.hosvc.test";


struct OPTIMIZE : public ::testing::Test, public DistrTestcase
{
protected:
	distr::iDistrMgrptrT make_mgr (const std::string& id)
	{
		return make_mgr(id, reserve_port());
	}

	distr::iDistrMgrptrT make_mgr (const std::string& id, size_t port)
	{
		return DistrTestcase::make_mgr(port, {
			distr::register_iosvc,
			distr::register_hosvc,
			distr::register_printsvc,
		}, id);
	}
};


TEST_F(OPTIMIZE, LocalCstrules)
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
	distr::iDistrMgrptrT mgr(make_mgr("mgr1"));

	teq::TensptrT rhs(eteq::make_functor(egen::ADD, {a, b}));

	auto& svc = distr::get_iosvc(*mgr);
	std::string id = svc.expose_node(rhs);

	// instance 2
	distr::iDistrMgrptrT mgr2 = make_mgr("mgr2");
	auto& svc2 = distr::get_iosvc(*mgr2);

	error::ErrptrT err = nullptr;
	teq::TensptrT rhs_ref = svc2.lookup_node(err, id);
	ASSERT_NOERR(err);
	teq::TensptrT lhs(eteq::make_functor(egen::ADD, {b, c}));
	teq::TensptrT everyone(eteq::make_functor(egen::ADD, {lhs, rhs_ref}));

	std::stringstream ss;
	distr::get_printsvc(*mgr2).print_ascii(ss, everyone.get());
	std::string expect =
		"(ADD)\n"
		"_`--(ADD)\n"
		"_|___`--(constant:[22\\15\\74\\38\\61\\...])\n"
		"_|___`--(constant:4)\n"
		"_`--[mgr1]:(ADD)\n"
		"_____`--(variable:a)\n"
		"_____`--(constant:[22\\15\\74\\38\\61\\...])\n";
	EXPECT_STREQ(expect.c_str(), ss.str().c_str());

	opt::Optimization optimization;
	auto optres = distr::get_hosvc(*mgr2).optimize({everyone}, optimization);
	ASSERT_EQ(1, optres.size());
	everyone = optres.front();

	std::stringstream ss2;
	distr::get_printsvc(*mgr2).print_ascii(ss2, everyone.get());
	std::string expect2 =
		"(ADD)\n"
		"_`--(constant:[26\\19\\78\\42\\65\\...])\n"
		"_`--[mgr1]:(ADD)\n"
		"_____`--(variable:a)\n"
		"_____`--(constant:[22\\15\\74\\38\\61\\...])\n";
	EXPECT_STREQ(expect2.c_str(), ss2.str().c_str());
}


TEST_F(OPTIMIZE, RemoteCstrules)
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
	distr::iDistrMgrptrT mgr(make_mgr("mgr1"));

	teq::TensptrT lhs(eteq::make_functor(egen::ADD, {b, c}));

	auto& svc = distr::get_iosvc(*mgr);
	std::string id = svc.expose_node(lhs);

	// instance 2
	distr::iDistrMgrptrT mgr2 = make_mgr("mgr2");
	auto& svc2 = distr::get_iosvc(*mgr2);

	error::ErrptrT err = nullptr;
	teq::TensptrT lhs_ref = svc2.lookup_node(err, id);
	ASSERT_NOERR(err);
	teq::TensptrT rhs(eteq::make_functor(egen::ADD, {a, b}));
	teq::TensptrT everyone(eteq::make_functor(egen::ADD, {lhs_ref, rhs}));

	std::stringstream ss;
	distr::get_printsvc(*mgr2).print_ascii(ss, everyone.get());
	std::string expect =
		"(ADD)\n"
		"_`--[mgr1]:(ADD)\n"
		"_|___`--(constant:[22\\15\\74\\38\\61\\...])\n"
		"_|___`--(constant:4)\n"
		"_`--(ADD)\n"
		"_____`--(variable:a)\n"
		"_____`--(constant:[22\\15\\74\\38\\61\\...])\n";
	EXPECT_STREQ(expect.c_str(), ss.str().c_str());

	opt::Optimization optimization;
	auto optres = distr::get_hosvc(*mgr2).optimize({everyone}, optimization);
	ASSERT_EQ(1, optres.size());
	everyone = optres.front();

	std::stringstream ss2;
	distr::get_printsvc(*mgr2).print_ascii(ss2, everyone.get());
	std::string expect2 =
		"(ADD)\n"
		"_`--[mgr1]:(constant:[26\\19\\78\\42\\65\\...])\n"
		"_`--(ADD)\n"
		"_____`--(variable:a)\n"
		"_____`--(constant:[22\\15\\74\\38\\61\\...])\n";
	EXPECT_STREQ(expect2.c_str(), ss2.str().c_str());
}


#endif // DISABLE_HOSVC_OPTIMIZE_TEST
