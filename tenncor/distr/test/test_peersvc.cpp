
#ifndef DISABLE_DISTR_PEERSVC_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/distr/mock/mock.hpp"


struct PEERSVC : public ::testing::Test, public DistrTestcase {};


TEST_F(PEERSVC, ManageClient)
{
	size_t port = reserve_port();
	std::string test_svc = "Basic";
	auto manager = DistrTestcase::make_mgr(port, {
		[](estd::ConfigMap<>& svcs, const distr::PeerServiceConfig& cfg) -> error::ErrptrT
		{
			svcs.add_entry<MockService>("test_mock_service",
				[&]{ return new MockService(cfg); });
			return nullptr;
		},
	}, test_svc);

	auto svc = dynamic_cast<MockService*>(manager->get_service("test_mock_service"));
	ASSERT_NE(nullptr, svc);

	error::ErrptrT err = nullptr;
	svc->public_client(err, test_svc);
	auto err_msg = fmts::sprintf(
		"cannot get client for local server %s", test_svc.c_str());
	EXPECT_ERR(err, err_msg.c_str());

	err = nullptr;
	svc->public_client(err, "clearly doesn't exist");
	char err_msg2[] = "cannot find client clearly doesn't exist";
	EXPECT_ERR(err, err_msg2);
}


#endif // DISABLE_DISTR_PEERSVC_TEST
