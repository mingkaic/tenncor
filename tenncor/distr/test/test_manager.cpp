
#ifndef DISABLE_DISTR_MANAGER_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/distr/distr.hpp"


struct MockClient
{
	MockClient (std::shared_ptr<grpc::Channel> channel,
		const egrpc::ClientConfig& cfg,
		const std::string& alias) {}
};


struct MockService : public distr::PeerService<MockClient>
{
	MockService (const distr::PeerServiceConfig& cfg) :
		PeerService<MockClient>(cfg) {}

	void register_service (grpc::ServerBuilder& builder) override
	{
		++registry_count_;
	}

	void initialize_server_call (grpc::ServerCompletionQueue& cq) override
	{
		++initial_count_;
	}

	size_t registry_count_ = 0;

	size_t initial_count_ = 0;
};


void register_mocksvc (estd::ConfigMap<>& svcs,
	distr::ConsulService* consulsvc)
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
	size_t test_port = 5112;
	std::string test_svc = "Empty";

	const char* consul_addr = std::getenv("TEST_CONSUL_ADDRESS");
	if (nullptr == consul_addr || 0 == std::strlen(consul_addr))
	{
		consul_addr = "localhost";
	}
	std::string address = fmts::sprintf("http://%s:8500", consul_addr);
	auto consul = std::make_shared<ppconsul::Consul>(address);
	auto consulsvc = distr::make_consul(consul, test_port, test_svc, "NoServices");

	estd::ConfigMap<> svcs;
	distr::DistrManager manager(distr::ConsulSvcptrT(consulsvc), svcs);
	EXPECT_STREQ("NoServices", manager.get_id().c_str());
	EXPECT_EQ(nullptr, manager.get_service("test_mock_service"));
}


TEST(MANAGER, Basic)
{
	size_t test_port = 5112;
	std::string test_svc = "Basic";

	const char* consul_addr = std::getenv("TEST_CONSUL_ADDRESS");
	if (nullptr == consul_addr || 0 == std::strlen(consul_addr))
	{
		consul_addr = "localhost";
	}
	std::string address = fmts::sprintf("http://%s:8500", consul_addr);
	auto consul = std::make_shared<ppconsul::Consul>(address);
	auto consulsvc = distr::make_consul(consul, test_port, test_svc, "Basic");

	estd::ConfigMap<> svcs;
	register_mocksvc(svcs, consulsvc);
	distr::DistrManager manager(distr::ConsulSvcptrT(consulsvc), svcs);
	EXPECT_STREQ("Basic", manager.get_id().c_str());

	auto svc = dynamic_cast<MockService*>(manager.get_service("test_mock_service"));
	EXPECT_NE(nullptr, svc);

	EXPECT_EQ(1, svc->registry_count_);
	EXPECT_EQ(1, svc->initial_count_);
}


#endif // DISABLE_DISTR_MANAGER_TEST
