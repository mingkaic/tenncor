
#ifndef DISABLE_DISTRIB_TEST

#include <cstdlib>

#include "testutil/tutil.hpp"

#include "exam/exam.hpp"

#include "distrib/manager.hpp"

#include "eteq/make.hpp"

#include "tenncor/tenncor.hpp"


const std::string test_service = "tenncor_distrib_ctest";


ppconsul::Consul create_test_consul (void)
{
	const char* consul_addr = std::getenv("TEST_CONSUL_ADDRESS");
	if (nullptr == consul_addr || 0 == std::strlen(consul_addr))
	{
		consul_addr = "localhost";
	}
	std::string address = fmts::sprintf("http://%s:8500", consul_addr);
	return ppconsul::Consul{address};
}


struct DISTRIB : public ::testing::Test
{
	DISTRIB (void) : consul_(create_test_consul()) {}

protected:
	void TearDown (void) override
	{
		ppconsul::agent::Agent agent(consul_);
		ppconsul::catalog::Catalog catalog(consul_);
		auto services = catalog.service(test_service);
		for (auto& service : services)
		{
			agent.deregisterService(service.second.id);
		}
	}

	void check_clean (void)
	{
		ppconsul::catalog::Catalog catalog(consul_);
		auto services = catalog.service(test_service);
		ASSERT_EQ(services.size(), 0);
	}

	distr::iDistMgrptrT make_mgr (size_t port, const std::string& id = "")
	{
		return std::make_shared<tcr::TenncorManager>(consul_, port,
			test_service, id, distr::ClientConfig(
				std::chrono::milliseconds(5000),
				std::chrono::milliseconds(10000),
				5
			));
	}

	ppconsul::Consul consul_;
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
		distr::iDistMgrptrT mgr = make_mgr(5112, "mgr1");
		tcr::set_distmgr(mgr);

		eteq::ETensor<double> src =
			eteq::make_constant<double>(data.data(), shape);
		eteq::ETensor<double> src2 =
			eteq::make_constant<double>(data2.data(), shape);
		eteq::ETensor<double> dest = src + src2;
		tcr::expose_node(dest);
		std::string id = tcr::lookup_id(dest);

		// instance 2
		distr::iDistMgrptrT mgr2 = make_mgr(5113, "mgr2");
		tcr::set_distmgr(mgr2);

		eteq::ETensor<double> src3 =
			eteq::make_constant<double>(data3.data(), shape);
		error::ErrptrT err = nullptr;
		auto ref = tcr::try_lookup_node<double>(err, id);
		ASSERT_NOERR(err);
		ASSERT_NE(nullptr, ref);
		auto dest2 = eteq::ETensor<double>(ref) * src3;

		tcr::set_distmgr(nullptr);
		eteq::ETensor<double> src4 =
			eteq::make_constant<double>(data.data(), shape);
		tcr::try_lookup_id(err, src4);
		EXPECT_ERR(err, "can only find reference ids using iDistManager");
	}
	check_clean();
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
		distr::iDistMgrptrT mgr = make_mgr(5112, "mgr1");
		tcr::set_distmgr(mgr);

		eteq::ETensor<double> src =
			eteq::make_constant<double>(data.data(), shape);
		eteq::ETensor<double> src2 =
			eteq::make_constant<double>(data2.data(), shape);
		eteq::ETensor<double> dest = src + src2;
		tcr::expose_node(dest);
		std::string id = tcr::lookup_id(dest);

		// instance 2
		distr::iDistMgrptrT mgr2 = make_mgr(5113, "mgr2");
		tcr::set_distmgr(mgr2);

		eteq::ETensor<double> src3 =
			eteq::make_constant<double>(data3.data(), shape);
		error::ErrptrT err = nullptr;
		auto ref = tcr::try_lookup_node<double>(err, id);
		ASSERT_NOERR(err);
		ASSERT_NE(nullptr, ref);
		auto dest2 = eteq::ETensor<double>(ref) * src3;

		ASSERT_TRUE(mgr->get_remotes().empty());
		ASSERT_EQ(mgr2->get_remotes().size(), 1);

		auto gotshape = dest2->shape();
		ASSERT_ARREQ(shape, gotshape);

		double* goptr = dest2.calc();
		for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(exdata[i], goptr[i]);
		}

		tcr::set_distmgr(nullptr);
	}
	check_clean();
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
		distr::iDistMgrptrT mgrA = make_mgr(5112, "mgrA");

		// instance B
		distr::iDistMgrptrT mgrB = make_mgr(5113, "mgrB");

		// instance C
		distr::iDistMgrptrT mgrC = make_mgr(5114, "mgrC");

		// instance D
		distr::iDistMgrptrT mgrD = make_mgr(5115, "mgrD");

		// instance E
		distr::iDistMgrptrT mgrE = make_mgr(5116, "mgrE");

		// instance F
		distr::iDistMgrptrT mgrF = make_mgr(5117, "mgrF");

		auto f1 = eteq::make_constant<double>(data.data(), shape);
		auto f2 = eteq::make_constant<double>(data2.data(), shape);
		auto d1 = eteq::make_constant<double>(data3.data(), shape);

		auto e2 = eteq::make_constant<double>(data.data(), shape);
		auto e3 = eteq::make_constant<double>(data2.data(), shape);
		auto e1 = e2 + e3;

		mgrE->expose_node(e1);
		mgrF->expose_node(f1);
		mgrF->expose_node(f2);
		mgrD->expose_node(d1);

		std::string e1_key = *mgrE->lookup_id(e1.get());

		error::ErrptrT err = nullptr;
		std::string f1_key = *mgrF->lookup_id(f1.get());
		eteq::ETensor<double> f1_ref = mgrA->lookup_node(err, f1_key);
		ASSERT_NOERR(err);
		auto a2 = -f1_ref;
		mgrA->expose_node(a2);

		std::string d1_key = *mgrD->lookup_id(d1.get());
		eteq::ETensor<double> a2_ref = mgrC->lookup_node(err, *mgrA->lookup_id(a2.get()));
		ASSERT_NOERR(err);
		eteq::ETensor<double> d1_ref = mgrC->lookup_node(err, d1_key);
		ASSERT_NOERR(err);
		auto c1 = a2_ref + d1_ref;
		mgrC->expose_node(c1);

		eteq::ETensor<double> c1_ref = mgrB->lookup_node(err, *mgrC->lookup_id(c1.get()));
		ASSERT_NOERR(err);
		eteq::ETensor<double> f2_ref = mgrB->lookup_node(err, *mgrF->lookup_id(f2.get()));
		ASSERT_NOERR(err);
		auto b1 = c1_ref * f2_ref;
		mgrB->expose_node(b1);

		eteq::ETensor<double> b1_ref = mgrA->lookup_node(err, *mgrB->lookup_id(b1.get()));
		ASSERT_NOERR(err);
		auto a1 = tenncor<double>().sin(b1_ref);
		mgrA->expose_node(a1);

		// able to reach across cyclical graph
		auto reachables = mgrA->find_reachable(err, {a1.get()}, {f1_key});
		EXPECT_EQ(1, reachables.size());
		EXPECT_EQ(a1.get(), *reachables.begin());

		// unable to reach isolated nodes
		auto nonreachable = mgrA->find_reachable(err, {a1.get()}, {e1_key});
		EXPECT_TRUE(nonreachable.empty());

		// unable to reach node unreachable nodes
		auto nonreachable2 = mgrA->find_reachable(err, {a2.get()}, {d1_key});
		EXPECT_TRUE(nonreachable2.empty());
	}
	check_clean();
}


TEST_F(DISTRIB, DISABLED_RemoteDeriving)
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
		std::vector<double> exdata = {
			63, 19,
			11, 94,
			23, 63,
		};

		// instance 1
		distr::iDistMgrptrT mgr = make_mgr(5112, "mgr1");
		tcr::set_distmgr(mgr);

		eteq::EVariable<double> a = eteq::make_variable<double>(data.data(), ashape);
		eteq::EVariable<double> b = eteq::make_variable<double>(data2.data(), bshape);
		eteq::EVariable<double> c = eteq::make_variable<double>(data3.data(), cshape);

		auto d = tenncor<double>().matmul(a, b);
		auto e = tenncor<double>().matmul(c, d);
		auto f = tenncor<double>().matmul(
			tenncor<double>().transpose(d),
			tenncor<double>().transpose(c));
		auto root = tenncor<double>().matmul(e, f);

		tcr::expose_node(root);
		tcr::expose_node(a);
		std::string root_id = tcr::lookup_id(root);
		std::string base_id = tcr::lookup_id(a);

		// instance 2
		distr::iDistMgrptrT mgr2 = make_mgr(5113, "mgr2");
		tcr::set_distmgr(mgr2);

		error::ErrptrT err = nullptr;
		eteq::ETensor<double> root_ref = tcr::try_lookup_node<double>(err, root_id);
		ASSERT_NOERR(err);
		eteq::ETensor<double> base_ref = tcr::try_lookup_node<double>(err, base_id);
		ASSERT_NOERR(err);

		auto dbases = tcr::derive(root_ref, {base_ref});
		ASSERT_EQ(1, dbases.size());
		auto dbase = dbases.front();

		auto gotshape = dbase->shape();
		ASSERT_ARREQ(ashape, gotshape);

		auto goptr = dbase.calc();
		for (size_t i = 0, n = gotshape.n_elems(); i < n; ++i)
		{
			EXPECT_DOUBLE_EQ(exdata[i], goptr[i]);
		}

		tcr::set_distmgr(nullptr);
	}
	check_clean();
}


#endif // DISABLE_DISTRIB_TEST
