
#ifndef DISABLE_DISTR_MANAGER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/distr/mock/mock.hpp"


void register_mocksvc (estd::ConfigMap<>& svcs,
	distr::iP2PService* consulsvc)
{
	distr::PeerServiceConfig cfg(consulsvc, egrpc::ClientConfig(
		std::chrono::milliseconds(5000),
		std::chrono::milliseconds(10000),
		5
	));
	svcs.add_entry<MockService>("test_mock_service",
		[&](){ return new MockService(cfg); });
}


TEST(MANAGER, NoServices)
{
	std::string test_svc = "NoServices";
	std::mutex kv_mtx;
	types::StrUMapT<std::string> kv;
	types::StrUMapT<std::string> peers = {
		{test_svc, "0.0.0.0:5112"},
	};

	estd::ConfigMap<> svcs;
	distr::DistrManager manager(
		std::make_unique<MockP2P>(kv_mtx, test_svc, peers, kv), svcs);
	EXPECT_STREQ("NoServices", manager.get_id().c_str());
	EXPECT_EQ(nullptr, manager.get_service("test_mock_service"));
}


TEST(MANAGER, Basic)
{
	std::string test_svc = "Basic";
	std::mutex kv_mtx;
	types::StrUMapT<std::string> kv;
	types::StrUMapT<std::string> peers = {
		{test_svc, "0.0.0.0:5112"},
	};
	auto consul_svc = new MockP2P(kv_mtx, test_svc, peers, kv);

	estd::ConfigMap<> svcs;
	register_mocksvc(svcs, consul_svc);
	distr::DistrManager manager(distr::P2PSvcptrT(consul_svc), svcs);
	EXPECT_STREQ("Basic", manager.get_id().c_str());

	auto svc = dynamic_cast<MockService*>(manager.get_service("test_mock_service"));
	EXPECT_NE(nullptr, svc);

	EXPECT_EQ(1, svc->registry_count_);
	EXPECT_EQ(1, svc->initial_count_);
}


#endif // DISABLE_DISTR_MANAGER_TEST
