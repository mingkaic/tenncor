
#ifndef DISABLE_DISTR_MANAGER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/distr/mock/mock.hpp"


struct MANAGER : public ::testing::Test, public DistrTestcase {};


TEST_F(MANAGER, SetGet)
{
	global::CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();

	size_t port = reserve_port();
	size_t port2 = reserve_port();
	std::string test_svc = "SetGet";
	std::string test_svc2 = "SetGet2";
	auto manager = DistrTestcase::make_mgr(port, {}, test_svc);
	auto manager2 = DistrTestcase::make_mgr(port2, {}, test_svc2);

	EXPECT_EQ(nullptr, distr::get_distrmgr(nullptr));
	EXPECT_EQ(nullptr, distr::get_distrmgr(ctx));

	distr::set_distrmgr(manager, nullptr);

	distr::set_distrmgr(nullptr, ctx);
	EXPECT_EQ(nullptr, distr::get_distrmgr(ctx));

	distr::set_distrmgr(manager, ctx);
	EXPECT_EQ(manager.get(), distr::get_distrmgr(ctx));

	distr::set_distrmgr(manager2, ctx);
	EXPECT_EQ(manager2.get(), distr::get_distrmgr(ctx));
}


TEST_F(MANAGER, NoServices)
{
	size_t port = reserve_port();
	std::string test_svc = "NoServices";
	auto manager = DistrTestcase::make_mgr(port, {}, test_svc);

	EXPECT_STREQ(test_svc.c_str(), manager->get_id().c_str());
	EXPECT_EQ(nullptr, manager->get_service("test_mock_service"));

	auto dmanager = dynamic_cast<distr::DistrManager*>(manager.get());
	ASSERT_NE(nullptr, dmanager);
	auto p2p = dynamic_cast<MockP2P*>(dmanager->get_p2psvc());
	ASSERT_NE(nullptr, p2p);

	EXPECT_STREQ(test_svc.c_str(), p2p->get_local_peer().c_str());
	EXPECT_STREQ(
		fmts::sprintf("0.0.0.0:%d", port).c_str(),
		p2p->get_local_addr().c_str());
	EXPECT_EQ(0, p2p->get_peers().size());
}


TEST_F(MANAGER, Basic)
{
	size_t port = reserve_port();
	std::string test_svc = "Basic";
	auto manager = DistrTestcase::make_mgr(port, {
		[](estd::ConfigMap<>& svcs, const distr::PeerServiceConfig& cfg) -> error::ErrptrT
		{
			svcs.add_entry<MockService>("test_mock_service",
				[&](){ return new MockService(cfg); });
			return nullptr;
		},
	}, test_svc);

	EXPECT_STREQ(test_svc.c_str(), manager->get_id().c_str());
	auto svc = dynamic_cast<MockService*>(manager->get_service("test_mock_service"));
	ASSERT_NE(nullptr, svc);
	EXPECT_EQ(1, svc->registry_count_);
	EXPECT_EQ(1, svc->initial_count_);

	auto dmanager = dynamic_cast<distr::DistrManager*>(manager.get());
	ASSERT_NE(nullptr, dmanager);
	auto p2p = dynamic_cast<MockP2P*>(dmanager->get_p2psvc());
	ASSERT_NE(nullptr, p2p);

	EXPECT_STREQ(test_svc.c_str(), p2p->get_local_peer().c_str());
	EXPECT_STREQ(
		fmts::sprintf("0.0.0.0:%d", port).c_str(),
		p2p->get_local_addr().c_str());
	EXPECT_EQ(0, p2p->get_peers().size());
}


#endif // DISABLE_DISTR_MANAGER_TEST
