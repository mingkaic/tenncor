
#ifndef DISABLE_DISTR_MANAGER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/distr/mock/mock.hpp"


struct MANAGER : public ::testing::Test, public DistrTestcase {};


TEST_F(MANAGER, NoServices)
{
	size_t port = reserve_port();
	std::string test_svc = "NoServices";
	auto manager = DistrTestcase::make_mgr(port, {}, test_svc);

	EXPECT_STREQ(test_svc.c_str(), manager->get_id().c_str());
	EXPECT_EQ(nullptr, manager->get_service("test_mock_service"));
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
	EXPECT_NE(nullptr, svc);
	EXPECT_EQ(1, svc->registry_count_);
	EXPECT_EQ(1, svc->initial_count_);
}


#endif // DISABLE_DISTR_MANAGER_TEST
