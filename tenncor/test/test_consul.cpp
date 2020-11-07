
#ifndef DISABLE_TENNCOR_CONSUL_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "tenncor/distr/mock/mock.hpp"


struct CONSUL : public ::testing::Test, public DistrTestcase
{
	void TearDown (void) override
	{
		ppconsul::Consul consul(consul_addr);

		ppconsul::agent::Agent agent(consul);
		ppconsul::kv::Kv kv(consul);

		agent.deregisterService(test_svc);
		kv.clear();
	}

	const std::string test_svc = "ConsulTest";

	const std::string consul_addr = "0.0.0.0:8500";
};


TEST_F(CONSUL, GetPeer)
{
	auto consul = std::make_shared<ppconsul::Consul>(consul_addr);

	size_t port = reserve_port();
	size_t port2 = reserve_port();
	std::string svc_id = "svc1";
	std::string svc2_id = "svc2";

	std::shared_ptr<distr::ConsulService> svc(distr::make_consul(
		consul, port, test_svc, "svc1"));
	distr::ConsulService svc2(consul, port2, "svc2", test_svc);

	auto peers = svc->get_peers();
	auto peers2 = svc2.get_peers();

	EXPECT_EQ(1, peers.size());
	EXPECT_EQ(1, peers2.size());

	std::string addr1 = fmts::sprintf("0.0.0.0:%d", port);
	std::string addr2 = fmts::sprintf("0.0.0.0:%d", port2);

	auto got_svcid = svc->get_local_peer();
	auto got_svcaddr = svc->get_local_addr();
	EXPECT_STREQ(svc_id.c_str(), got_svcid.c_str());
	EXPECT_STREQ(addr1.c_str(), got_svcaddr.c_str());

	auto it = peers.begin();
	EXPECT_STREQ(svc2_id.c_str(), it->first.c_str());
	EXPECT_STREQ(addr2.c_str(), it->second.c_str());

	auto got_svc2id = svc2.get_local_peer();
	auto got_svc2addr = svc2.get_local_addr();
	EXPECT_STREQ(svc2_id.c_str(), got_svc2id.c_str());
	EXPECT_STREQ(addr2.c_str(), got_svc2addr.c_str());

	auto it2 = peers2.begin();
	EXPECT_STREQ(svc_id.c_str(), it2->first.c_str());
	EXPECT_STREQ(addr1.c_str(), it2->second.c_str());
}


TEST_F(CONSUL, SetGetKv)
{
	auto consul = std::make_shared<ppconsul::Consul>(consul_addr);

	size_t port = reserve_port();
	std::string svc_id = "svc1";

	distr::ConsulService svc(consul, port, test_svc, "svc1");

	auto defval = svc.get_kv("abc", "def");
	EXPECT_STREQ("def", defval.c_str());

	svc.set_kv("abc", "ghi");

	auto gotval = svc.get_kv("abc", "def");
	EXPECT_STREQ("ghi", gotval.c_str());
}


#endif // DISABLE_TENNCOR_CONSUL_TEST
