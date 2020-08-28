
#ifndef DISABLE_DISTRIB_TEST

#include <cstdlib>

#include "testutil/tutil.hpp"

#include "exam/exam.hpp"

#include "dbg/print/printsvc/service.hpp"
#include "tenncor/distrib/manager.hpp"

#include "tenncor/eteq/make.hpp"

#include "tenncor/tenncor.hpp"


const std::string test_service = "tenncor_distrib_ctest";


struct DISTRIB : public ::testing::Test
{
	DISTRIB (void)
	{
		const char* consul_addr = std::getenv("TEST_CONSUL_ADDRESS");
		if (nullptr == consul_addr || 0 == std::strlen(consul_addr))
		{
			consul_addr = "localhost";
		}
		std::string address = fmts::sprintf("http://%s:8500", consul_addr);
		consul_ = std::make_shared<ppconsul::Consul>(address);
		clean_up();
	}

	~DISTRIB (void)
	{
		clean_up();
	}

protected:
	void TearDown (void) override
	{
		clean_up();
	}

	distr::iDistrMgrptrT make_mgr (size_t port, const std::string& id = "",
		global::CfgMapptrT ctx = global::context())
	{
		return tcr::ctxualize_distrmgr(consul_, port, id, {
			distr::register_iosvc,
			distr::register_opsvc,
			distr::register_oxsvc,
			distr::register_printsvc,
		}, test_service, ctx);
	}

	void clean_up (void)
	{
		ppconsul::agent::Agent agent(*consul_);
		ppconsul::catalog::Catalog catalog(*consul_);
		auto services = catalog.service(test_service);
		for (auto& service : services)
		{
			agent.deregisterService(service.second.id);
		}
	}

	void check_clean (void)
	{
		ppconsul::catalog::Catalog catalog(*consul_);
		auto services = catalog.service(test_service);
		ASSERT_EQ(services.size(), 0);
	}

	distr::ConsulptrT consul_;
};


TEST_F(DISTRIB, SharingNodes)
{
	{
		teq::Shape shape({2, 3, 4});
		std::vector<double> data = {
			63, 19, 11, 94, 23, 63,
			3, 48, 60, 77, 62, 32,
			35, 89, 33, 64, 36, 64,
			25, 49, 41, 1, 4, 97,
		};
		std::vector<double> data2 = {
			18, 30, 23, 60, 36, 60,
			73, 36, 6, 66, 67, 84,
			54, 43, 29, 8, 20, 71,
			10, 53, 90, 7, 94, 87,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69, 84, 67,
			66, 59, 69, 92, 96, 18,
			55, 35, 40, 81, 40, 18,
			60, 70, 68, 65, 30, 25,
		};

		// instance 1
		auto mgr = make_mgr(5112, "mgr1");

		eteq::ETensor src =
			eteq::make_constant<double>(data.data(), shape);
		eteq::ETensor src2 =
			eteq::make_constant<double>(data2.data(), shape);
		eteq::ETensor dest = src + src2;
		tcr::expose_node(dest);
		std::string id = tcr::lookup_id(dest);

		// instance 2
		auto mgr2 = make_mgr(5113, "mgr2");

		eteq::ETensor src3 =
			eteq::make_constant<double>(data3.data(), shape);
		error::ErrptrT err = nullptr;
		auto ref = tcr::try_lookup_node(err, id);
		ASSERT_NOERR(err);
		ASSERT_NE(nullptr, ref);
		auto dest2 = eteq::ETensor(ref) * src3;

		tcr::set_distrmgr(nullptr);
		eteq::ETensor src4 =
			eteq::make_constant<double>(data.data(), shape);
		tcr::try_lookup_id(err, src4);
		EXPECT_ERR(err, "can only find reference ids using DistrManager");
	}
	check_clean();
	auto global = global::context();
	EXPECT_EQ(nullptr, tcr::get_distrmgr(global));
	EXPECT_NE(nullptr, dynamic_cast<teq::Evaluator*>(&teq::get_eval(global)));
}


TEST_F(DISTRIB, DataPassing)
{
	{
		eigen::Device device;
		teq::Shape shape({2, 3, 4});
		std::vector<double> data = {
			63, 19, 11, 94, 23, 63,
			3, 48, 60, 77, 62, 32,
			35, 89, 33, 64, 36, 64,
			25, 49, 41, 1, 4, 97,
		};
		std::vector<double> data2 = {
			18, 30, 23, 60, 36, 60,
			73, 36, 6, 66, 67, 84,
			54, 43, 29, 8, 20, 71,
			10, 53, 90, 7, 94, 87,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69, 84, 67,
			66, 59, 69, 92, 96, 18,
			55, 35, 40, 81, 40, 18,
			60, 70, 68, 65, 30, 25,
		};
		std::vector<double> exdata = {
			2997, 3430, 68, 10626, 4956, 8241,
			5016, 4956, 4554, 13156, 12384, 2088,
			4895, 4620, 2480, 5832, 2240, 2430,
			2100, 7140, 8908, 520, 2940, 4600,
		};

		// instance 1
		auto mgr = make_mgr(5112, "mgr1");

		eteq::ETensor src =
			eteq::make_constant<double>(data.data(), shape);
		eteq::ETensor src2 =
			eteq::make_constant<double>(data2.data(), shape);
		eteq::ETensor dest = src + src2;
		tcr::expose_node(dest);
		std::string id = tcr::lookup_id(dest);

		// instance 2
		auto mgr2 = make_mgr(5113, "mgr2");

		eteq::ETensor src3 =
			eteq::make_constant<double>(data3.data(), shape);
		error::ErrptrT err = nullptr;
		auto ref = tcr::try_lookup_node(err, id);
		ASSERT_NOERR(err);
		ASSERT_NE(nullptr, ref);
		auto dest2 = eteq::ETensor(ref) * src3;

		ASSERT_TRUE(distr::get_iosvc(*mgr).get_remotes().empty());
		ASSERT_EQ(distr::get_iosvc(*mgr2).get_remotes().size(), 1);

		auto gotshape = dest2->shape();
		ASSERT_ARREQ(shape, gotshape);

		double* goptr = dest2.calc<double>();
		for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(exdata[i], goptr[i]);
		}

		tcr::set_distrmgr(nullptr);
	}
	check_clean();
	auto global = global::context();
	EXPECT_EQ(nullptr, tcr::get_distrmgr(global));
	EXPECT_NE(nullptr, dynamic_cast<teq::Evaluator*>(&teq::get_eval(global)));
}


TEST_F(DISTRIB, ComplexDataPassing)
{
	{
		eigen::Device device;
		teq::Shape shape({2, 3, 4});
		std::vector<double> data = {
			63, 19, 11, 94, 23, 63,
			3, 48, 60, 77, 62, 32,
			35, 89, 33, 64, 36, 64,
			25, 49, 41, 1, 4, 97,
		};
		std::vector<double> data2 = {
			18, 30, 23, 60, 36, 60,
			73, 36, 6, 66, 67, 84,
			54, 43, 29, 8, 20, 71,
			10, 53, 90, 7, 94, 87,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69, 84, 67,
			66, 59, 69, 92, 96, 18,
			55, 35, 40, 81, 40, 18,
			60, 70, 68, 65, 30, 25,
		};

		std::vector<double> exdata;
		{
			auto f1 = eteq::make_variable<double>(data.data(), shape, "f1");
			auto f2 = eteq::make_variable<double>(data2.data(), shape, "f2");
			auto d1 = eteq::make_variable<double>(data3.data(), shape, "d1");
			auto a2 = -f1;
			auto c1 = a2 + d1;
			auto b1 = c1 * f2;
			auto a1 = tenncor().sin(b1);

			auto exshape = a1->shape();
			ASSERT_ARREQ(shape, exshape);
			auto exptr = a1.template calc<double>();
			exdata = std::vector<double>(exptr, exptr + exshape.n_elems());
		}

		// instance A
		auto actx = std::make_shared<estd::ConfigMap<>>();
		auto mgrA = make_mgr(5112, "mgrA", actx);

		// instance B
		auto bctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrB = make_mgr(5113, "mgrB", bctx);

		// instance C
		auto cctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrC = make_mgr(5114, "mgrC", cctx);

		// instance D
		auto dctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrD = make_mgr(5115, "mgrD", dctx);

		// instance E
		auto ectx = std::make_shared<estd::ConfigMap<>>();
		auto mgrE = make_mgr(5116, "mgrE", ectx);

		// instance F
		auto fctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrF = make_mgr(5117, "mgrF", fctx);

		auto f1 = eteq::make_variable<double>(
			data.data(), shape, "f1", fctx);
		auto f2 = eteq::make_variable<double>(
			data2.data(), shape, "f2", fctx);

		auto d1 = eteq::make_variable<double>(
			data3.data(), shape, "d1", dctx);

		auto e2 = eteq::make_variable<double>(
			data.data(), shape, "e2", ectx);
		auto e3 = eteq::make_variable<double>(
			data2.data(), shape, "e3", ectx);
		auto e1 = TenncorAPI(ectx).add(e2, e3);

		tcr::expose_node(e1);
		tcr::expose_node(f1);
		tcr::expose_node(f2);
		tcr::expose_node(d1);

		error::ErrptrT err = nullptr;
		std::string f1_key = tcr::lookup_id(f1);
		auto f1_ref = tcr::lookup_node(f1_key, actx);
		ASSERT_NOERR(err);
		auto a2 = TenncorAPI(actx).neg(f1_ref);
		tcr::expose_node(a2);

		std::string d1_key = tcr::lookup_id(d1);
		auto a2_ref = tcr::lookup_node(tcr::lookup_id(a2), cctx);
		ASSERT_NOERR(err);
		auto d1_ref = tcr::lookup_node(d1_key, cctx);
		auto d1_ref_fromA = tcr::lookup_node(d1_key, actx);
		ASSERT_NOERR(err);
		auto c1 = TenncorAPI(cctx).add(a2_ref, d1_ref);
		tcr::expose_node(c1);

		auto c1_ref = tcr::lookup_node(tcr::lookup_id(c1), bctx);
		ASSERT_NOERR(err);
		auto f2_ref = tcr::lookup_node(tcr::lookup_id(f2), bctx);
		ASSERT_NOERR(err);
		auto b1 = TenncorAPI(bctx).mul(c1_ref, f2_ref);
		tcr::expose_node(b1);

		auto b1_ref = tcr::lookup_node(tcr::lookup_id(b1), actx);
		ASSERT_NOERR(err);
		auto a1 = TenncorAPI(actx).sin(b1_ref);

		auto gotshape = a1->shape();
		ASSERT_ARREQ(shape, gotshape);
		auto goptr = a1.template calc<double>();

		for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(exdata[i], goptr[i]);
		}
	}
	check_clean();
	auto global = global::context();
	EXPECT_EQ(nullptr, tcr::get_distrmgr(global));
	EXPECT_NE(nullptr, dynamic_cast<teq::Evaluator*>(&teq::get_eval(global)));
}


TEST_F(DISTRIB, Reachability)
{
	{
		eigen::Device device;
		teq::Shape shape({2, 3, 4});
		std::vector<double> data = {
			63, 19, 11, 94, 23, 63,
			3, 48, 60, 77, 62, 32,
			35, 89, 33, 64, 36, 64,
			25, 49, 41, 1, 4, 97,
		};
		std::vector<double> data2 = {
			18, 30, 23, 60, 36, 60,
			73, 36, 6, 66, 67, 84,
			54, 43, 29, 8, 20, 71,
			10, 53, 90, 7, 94, 87,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69, 84, 67,
			66, 59, 69, 92, 96, 18,
			55, 35, 40, 81, 40, 18,
			60, 70, 68, 65, 30, 25,
		};

		// instance A
		distr::iDistrMgrptrT mgrA(make_mgr(5112, "mgrA", nullptr));

		// instance B
		distr::iDistrMgrptrT mgrB(make_mgr(5113, "mgrB", nullptr));

		// instance C
		distr::iDistrMgrptrT mgrC(make_mgr(5114, "mgrC", nullptr));

		// instance D
		distr::iDistrMgrptrT mgrD(make_mgr(5115, "mgrD", nullptr));

		// instance E
		distr::iDistrMgrptrT mgrE(make_mgr(5116, "mgrE", nullptr));

		// instance F
		distr::iDistrMgrptrT mgrF(make_mgr(5117, "mgrF", nullptr));

		auto f1 = eteq::make_constant<double>(data.data(), shape);
		auto f2 = eteq::make_constant<double>(data2.data(), shape);
		auto d1 = eteq::make_constant<double>(data3.data(), shape);

		auto e2 = eteq::make_constant<double>(data.data(), shape);
		auto e3 = eteq::make_constant<double>(data2.data(), shape);
		auto e1 = e2 + e3;

		distr::get_iosvc(*mgrE).expose_node(e1);
		distr::get_iosvc(*mgrF).expose_node(f1);
		distr::get_iosvc(*mgrF).expose_node(f2);
		distr::get_iosvc(*mgrD).expose_node(d1);

		std::string e1_key = *distr::get_iosvc(*mgrE).lookup_id(e1.get());

		error::ErrptrT err = nullptr;
		std::string f1_key = *distr::get_iosvc(*mgrF).lookup_id(f1.get());
		eteq::ETensor f1_ref = distr::get_iosvc(*mgrA).lookup_node(err, f1_key);
		ASSERT_NOERR(err);
		auto a2 = -f1_ref;
		distr::get_iosvc(*mgrA).expose_node(a2);

		std::string d1_key = *distr::get_iosvc(*mgrD).lookup_id(d1.get());
		eteq::ETensor a2_ref = distr::get_iosvc(*mgrC).lookup_node(err, *distr::get_iosvc(*mgrA).lookup_id(a2.get()));
		ASSERT_NOERR(err);
		eteq::ETensor d1_ref = distr::get_iosvc(*mgrC).lookup_node(err, d1_key);
		ASSERT_NOERR(err);
		auto c1 = a2_ref + d1_ref;
		distr::get_iosvc(*mgrC).expose_node(c1);

		eteq::ETensor c1_ref = distr::get_iosvc(*mgrB).lookup_node(err, *distr::get_iosvc(*mgrC).lookup_id(c1.get()));
		ASSERT_NOERR(err);
		eteq::ETensor f2_ref = distr::get_iosvc(*mgrB).lookup_node(err, *distr::get_iosvc(*mgrF).lookup_id(f2.get()));
		ASSERT_NOERR(err);
		auto b1 = c1_ref * f2_ref;
		distr::get_iosvc(*mgrB).expose_node(b1);

		eteq::ETensor b1_ref = distr::get_iosvc(*mgrA).lookup_node(err, *distr::get_iosvc(*mgrB).lookup_id(b1.get()));
		ASSERT_NOERR(err);
		auto a1 = tenncor().sin(b1_ref);
		distr::get_iosvc(*mgrA).expose_node(a1);

		// able to reach across cyclical graph
		auto reachables = estd::map_keyset(distr::get_opsvc(*mgrA).reachable(err, {a1.get()}, {f1_key}));
		EXPECT_EQ(1, reachables.size());
		EXPECT_EQ(a1.get(), *reachables.begin());

		// unable to reach isolated nodes
		auto nonreachable = distr::get_opsvc(*mgrA).reachable(err, {a1.get()}, {e1_key});
		EXPECT_TRUE(nonreachable.empty());

		// unable to reach node unreachable nodes
		auto nonreachable2 = distr::get_opsvc(*mgrA).reachable(err, {a2.get()}, {d1_key});
		EXPECT_TRUE(nonreachable2.empty());
	}
	check_clean();
}


TEST_F(DISTRIB, RemoteDeriving)
{
	{
		eigen::Device device;
		teq::Shape ashape({2, 3});
		teq::Shape bshape({3, 4});
		teq::Shape cshape({4, 2});

		std::vector<double> data = {
			63, 19,
			11, 94,
			23, 63,
		};
		std::vector<double> data2 = {
			18, 30, 23,
			60, 36, 60,
			73, 36, 6,
			66, 67, 84,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69,
			84, 67, 66, 59,
		};

		std::vector<double> df_data;
		{
			eteq::EVariable<double> a = eteq::make_variable<double>(data.data(), ashape, "a");
			eteq::EVariable<double> b = eteq::make_variable<double>(data2.data(), bshape, "b");
			eteq::EVariable<double> c = eteq::make_variable<double>(data3.data(), cshape, "c");

			auto d = tenncor().matmul(b, a);
			auto e = tenncor().matmul(c, d);
			auto f = tenncor().matmul(
				tenncor().transpose(d),
				tenncor().transpose(c));
			auto root = tenncor().matmul(e, f);

			auto ders = tcr::derive(root, {a});
			auto exd = ders.front();

			auto exshape = exd->shape();
			ASSERT_ARREQ(ashape, exshape);
			auto exptr = exd.template calc<double>();
			df_data = std::vector<double>(exptr, exptr + exshape.n_elems());
		}

		// instance 1
		auto mgr = make_mgr(5112, "mgr1");

		eteq::EVariable<double> a = eteq::make_variable<double>(data.data(), ashape, "a");
		eteq::EVariable<double> b = eteq::make_variable<double>(data2.data(), bshape, "b");
		eteq::EVariable<double> c = eteq::make_variable<double>(data3.data(), cshape, "c");

		auto d = tenncor().matmul(b, a);
		auto e = tenncor().matmul(c, d);
		auto f = tenncor().matmul(
			tenncor().transpose(d),
			tenncor().transpose(c));
		auto root = tenncor().matmul(e, f);

		tcr::expose_node(root);
		tcr::expose_node(a);
		std::string root_id = tcr::lookup_id(root);
		std::string base_id = tcr::lookup_id(a);

		// instance 2
		auto mgr2 = make_mgr(5113, "mgr2");

		error::ErrptrT err = nullptr;
		eteq::ETensor root_ref = tcr::try_lookup_node(err, root_id);
		ASSERT_NOERR(err);
		eteq::ETensor base_ref = tcr::try_lookup_node(err, base_id);
		ASSERT_NOERR(err);

		auto remote_ders = tcr::derive(root_ref, {base_ref});
		ASSERT_EQ(1, remote_ders.size());
		auto dbase = remote_ders.front();

		auto gotshape = dbase->shape();
		ASSERT_ARREQ(ashape, gotshape);
		auto goptr = dbase.template calc<double>();

		for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(df_data[i], goptr[i]);
		}

		tcr::set_distrmgr(nullptr);
	}
	check_clean();
	auto global = global::context();
	EXPECT_EQ(nullptr, tcr::get_distrmgr(global));
	EXPECT_NE(nullptr, dynamic_cast<teq::Evaluator*>(&teq::get_eval(global)));
}


TEST_F(DISTRIB, DebugPrintAscii)
{
	{
		eigen::Device device;
		teq::Shape shape({2, 3, 4});
		std::vector<double> data = {
			63, 19, 11, 94, 23, 63,
			3, 48, 60, 77, 62, 32,
			35, 89, 33, 64, 36, 64,
			25, 49, 41, 1, 4, 97,
		};
		std::vector<double> data2 = {
			18, 30, 23, 60, 36, 60,
			73, 36, 6, 66, 67, 84,
			54, 43, 29, 8, 20, 71,
			10, 53, 90, 7, 94, 87,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69, 84, 67,
			66, 59, 69, 92, 96, 18,
			55, 35, 40, 81, 40, 18,
			60, 70, 68, 65, 30, 25,
		};

		// instance A
		auto actx = std::make_shared<estd::ConfigMap<>>();
		auto mgrA = make_mgr(5112, "mgrA", actx);

		// instance B
		auto bctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrB = make_mgr(5113, "mgrB", bctx);

		// instance C
		auto cctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrC = make_mgr(5114, "mgrC", cctx);

		// instance D
		auto dctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrD = make_mgr(5115, "mgrD", dctx);

		// instance E
		auto ectx = std::make_shared<estd::ConfigMap<>>();
		auto mgrE = make_mgr(5116, "mgrE", ectx);

		// instance F
		auto fctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrF = make_mgr(5117, "mgrF", fctx);

		auto f1 = eteq::make_variable<double>(
			data.data(), shape, "f1", fctx);
		auto f2 = eteq::make_variable<double>(
			data2.data(), shape, "f2", fctx);

		auto d1 = eteq::make_variable<double>(
			data3.data(), shape, "d1", dctx);

		auto e2 = eteq::make_variable<double>(
			data.data(), shape, "e2", ectx);
		auto e3 = eteq::make_variable<double>(
			data2.data(), shape, "e3", ectx);
		auto e1 = TenncorAPI(ectx).add(e2, e3);

		tcr::expose_node(e1);
		tcr::expose_node(f1);
		tcr::expose_node(f2);
		tcr::expose_node(d1);

		error::ErrptrT err = nullptr;
		auto f1_ref = tcr::lookup_node(tcr::lookup_id(f1), actx);
		ASSERT_NOERR(err);
		auto a2 = TenncorAPI(actx).neg(f1_ref);
		tcr::expose_node(a2);

		auto a2_ref = tcr::lookup_node(tcr::lookup_id(a2), cctx);
		ASSERT_NOERR(err);
		auto d1_ref = tcr::lookup_node(tcr::lookup_id(d1), cctx);
		ASSERT_NOERR(err);
		auto c1 = TenncorAPI(cctx).add(a2_ref, d1_ref);
		tcr::expose_node(c1);

		auto c1_ref = tcr::lookup_node(tcr::lookup_id(c1), bctx);
		ASSERT_NOERR(err);
		auto f2_ref = tcr::lookup_node(tcr::lookup_id(f2), bctx);
		ASSERT_NOERR(err);
		auto b1 = TenncorAPI(bctx).mul(c1_ref, f2_ref);
		tcr::expose_node(b1);

		auto b1_ref = tcr::lookup_node(tcr::lookup_id(b1), actx);
		ASSERT_NOERR(err);
		auto a1 = TenncorAPI(actx).sin(b1_ref);
		tcr::expose_node(a1);

		// a1 -> b1 -> c1, f2
		// c1 -> a2, d1
		// a2 -> f1

		std::stringstream ss;
		distr::get_printsvc(*mgrA).print_ascii(ss, a1.get());

		std::string expect =
			"(SIN)\n"
			"_`--[mgrB]:(MUL)\n"
			"_____`--[mgrC]:(ADD)\n"
			"_____|___`--[mgrA]:(NEG)\n"
			"_____|___|___`--[mgrF]:(variable:f1)\n"
			"_____|___`--[mgrD]:(variable:d1)\n"
			"_____`--[mgrF]:(variable:f2)\n";
		EXPECT_STREQ(expect.c_str(), ss.str().c_str());
	}
	check_clean();
}


TEST_F(DISTRIB, CrossDerive)
{
	{
		eigen::Device device;
		teq::Shape shape({2, 3, 4});
		std::vector<double> data = {
			63, 19, 11, 94, 23, 63,
			3, 48, 60, 77, 62, 32,
			35, 89, 33, 64, 36, 64,
			25, 49, 41, 1, 4, 97,
		};
		std::vector<double> data2 = {
			18, 30, 23, 60, 36, 60,
			73, 36, 6, 66, 67, 84,
			54, 43, 29, 8, 20, 71,
			10, 53, 90, 7, 94, 87,
		};
		std::vector<double> data3 = {
			37, 70, 2, 69, 84, 67,
			66, 59, 69, 92, 96, 18,
			55, 35, 40, 81, 40, 18,
			60, 70, 68, 65, 30, 25,
		};

		std::vector<double> df_data;
		{
			auto f1 = eteq::make_variable<double>(data.data(), shape, "f1");
			auto f2 = eteq::make_variable<double>(data2.data(), shape, "f2");
			auto d1 = eteq::make_variable<double>(data3.data(), shape, "d1");
			auto a2 = -f1;
			auto c1 = a2 + d1;
			auto b1 = c1 * f2;
			auto a1 = tenncor().sin(b1);
			auto exds = tcr::derive(a1, {f1});
			ASSERT_EQ(1, exds.size());
			auto exd = exds.front();
			ASSERT_NE(nullptr, exd);
			exd = tenncor().cast<double>(exd);

			auto exshape = exd->shape();
			ASSERT_ARREQ(shape, exshape);
			auto exptr = exd.template calc<double>();
			df_data = std::vector<double>(exptr, exptr + exshape.n_elems());
		}

		// instance A
		auto actx = std::make_shared<estd::ConfigMap<>>();
		auto mgrA = make_mgr(5112, "mgrA", actx);

		// instance B
		auto bctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrB = make_mgr(5113, "mgrB", bctx);

		// instance C
		auto cctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrC = make_mgr(5114, "mgrC", cctx);

		// instance D
		auto dctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrD = make_mgr(5115, "mgrD", dctx);

		// instance E
		auto ectx = std::make_shared<estd::ConfigMap<>>();
		auto mgrE = make_mgr(5116, "mgrE", ectx);

		// instance F
		auto fctx = std::make_shared<estd::ConfigMap<>>();
		auto mgrF = make_mgr(5117, "mgrF", fctx);

		auto f1 = eteq::make_variable<double>(
			data.data(), shape, "f1", fctx);
		auto f2 = eteq::make_variable<double>(
			data2.data(), shape, "f2", fctx);

		auto d1 = eteq::make_variable<double>(
			data3.data(), shape, "d1", dctx);

		auto e2 = eteq::make_variable<double>(
			data.data(), shape, "e2", ectx);
		auto e3 = eteq::make_variable<double>(
			data2.data(), shape, "e3", ectx);
		auto e1 = TenncorAPI(ectx).add(e2, e3);

		tcr::expose_node(e1);
		tcr::expose_node(f1);
		tcr::expose_node(f2);
		tcr::expose_node(d1);

		error::ErrptrT err = nullptr;
		std::string f1_key = tcr::lookup_id(f1);
		auto f1_ref = tcr::lookup_node(f1_key, actx);
		ASSERT_NOERR(err);
		auto a2 = TenncorAPI(actx).neg(f1_ref);
		tcr::expose_node(a2);

		std::string d1_key = tcr::lookup_id(d1);
		auto a2_ref = tcr::lookup_node(tcr::lookup_id(a2), cctx);
		ASSERT_NOERR(err);
		auto d1_ref = tcr::lookup_node(d1_key, cctx);
		auto d1_ref_fromA = tcr::lookup_node(d1_key, actx);
		ASSERT_NOERR(err);
		auto c1 = TenncorAPI(cctx).add(a2_ref, d1_ref);
		tcr::expose_node(c1);

		auto c1_ref = tcr::lookup_node(tcr::lookup_id(c1), bctx);
		ASSERT_NOERR(err);
		auto f2_ref = tcr::lookup_node(tcr::lookup_id(f2), bctx);
		ASSERT_NOERR(err);
		auto b1 = TenncorAPI(bctx).mul(c1_ref, f2_ref);
		tcr::expose_node(b1);

		auto b1_ref = tcr::lookup_node(tcr::lookup_id(b1), actx);
		ASSERT_NOERR(err);
		auto a1 = TenncorAPI(actx).sin(b1_ref);
		tcr::expose_node(a1);

		// a1 -> b1 -> c1, f2
		// c1 -> a2, d1
		// a2 -> f1

		// expect derive path:
		// A -> grads={a1:[1],b1:[cos(b1)*1]} -> B
		// B -> grads={b1:[cos(b1)],c1:[f2*cos(b1)]} -> C
		// C -> grads={c1:[f2*cos(b1)],a2:[f2*cos(b1)]} -> A
		// A -> grads={a2:[f2*cos(b1)],f1:[-f2*cos(b1)]} -> F
		// F -> grads={f1:[-f2*cos(b1)]}

		auto e1_ref = tcr::lookup_node(tcr::lookup_id(e1), actx);
		auto ders = tcr::derive(a1, {f1_ref, e1_ref});
		ASSERT_EQ(2, ders.size());

		// able to derive across cyclical graph
		{
			auto df1 = ders[0];
			df1 = TenncorAPI(actx).cast<double>(df1);
			auto goptr = df1.template calc<double>();

			std::stringstream ss;
			distr::get_printsvc(*mgrA).print_ascii(ss, df1.get());
			std::string expect =
				"(CAST)\n"
				"_`--(NEG)\n"
				"_____`--[mgrB]:(MUL)\n"
				"_________`--[mgrF]:(variable:f2)\n"
				"_________`--[mgrA]:(MUL)\n"
				"_____________`--(COS)\n"
				"_____________|___`--[mgrB]:(MUL)\n"
				"_____________|_______`--[mgrC]:(ADD)\n"
				"_____________|_______|___`--[mgrA]:(NEG)\n"
				"_____________|_______|___|___`--[mgrF]:(variable:f1)\n"
				"_____________|_______|___`--[mgrD]:(variable:d1)\n"
				"_____________|_______`--[mgrF]:(variable:f2)\n"
				"_____________`--(CAST)\n"
				"_________________`--(constant:1)\n";
			EXPECT_STREQ(expect.c_str(), ss.str().c_str());

			auto gotshape = df1->shape();
			ASSERT_ARREQ(shape, gotshape);

			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				EXPECT_DOUBLE_EQ(df_data[i], goptr[i]);
			}
		}

		// unable to reach isolated nodes
		{
			auto de1 = ders[1];
			de1 = TenncorAPI(actx).cast<double>(de1);

			auto gotshape = de1->shape();
			ASSERT_ARREQ(shape, gotshape);
			auto goptr = de1.template calc<double>();

			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				EXPECT_DOUBLE_EQ(0, goptr[i]);
			}
		}

		// unable to reach node unreachable nodes
		{
			auto ders2 = tcr::derive(a2, {d1_ref_fromA});
			ASSERT_EQ(1, ders2.size());
			auto dd1 = ders2[0];
			dd1 = TenncorAPI(actx).cast<double>(dd1);

			auto gotshape = dd1->shape();
			ASSERT_ARREQ(shape, gotshape);
			auto goptr = dd1.template calc<double>();

			for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
			{
				EXPECT_DOUBLE_EQ(0, goptr[i]);
			}
		}
	}
	check_clean();
}


#endif // DISABLE_DISTRIB_TEST
