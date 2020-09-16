
#ifndef DISABLE_IOSVC_LOOKUP_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "testutil/tutil.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/distr.hpp"

#include "tenncor/distr/iosvc/iosvc.hpp"


TEST(LOOKUP, LookupId)
{
	size_t test_port = 5112;

	const char* consul_addr = std::getenv("TEST_CONSUL_ADDRESS");
	if (nullptr == consul_addr || 0 == std::strlen(consul_addr))
	{
		consul_addr = "localhost";
	}
	std::string address = fmts::sprintf("http://%s:8500", consul_addr);
	auto consul = std::make_shared<ppconsul::Consul>(address);
	auto consulsvc = distr::make_consul(consul, test_port, "Lookup.IOService", "service1");

	distr::PeerServiceConfig cfg(consulsvc, egrpc::ClientConfig(
		std::chrono::milliseconds(5000),
		std::chrono::milliseconds(10000),
		5
	));
	distr::io::DistrIOService service(cfg);

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	auto b = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	service.expose_node(a);

	auto ida = service.lookup_id(a.get());
	auto idb = service.lookup_id(b.get());
	EXPECT_TRUE(ida);
	EXPECT_FALSE(idb);
}


TEST(LOOKUP, LocalLookupNode)
{
	size_t test_port = 5112;

	const char* consul_addr = std::getenv("TEST_CONSUL_ADDRESS");
	if (nullptr == consul_addr || 0 == std::strlen(consul_addr))
	{
		consul_addr = "localhost";
	}
	std::string address = fmts::sprintf("http://%s:8500", consul_addr);
	auto consul = std::make_shared<ppconsul::Consul>(address);
	auto consulsvc = distr::make_consul(consul, test_port, "Lookup.IOService", "service1");

	distr::PeerServiceConfig cfg(consulsvc, egrpc::ClientConfig(
		std::chrono::milliseconds(5000),
		std::chrono::milliseconds(10000),
		5
	));
	distr::io::DistrIOService service(cfg);

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	service.expose_node(a);
	auto ida = *service.lookup_id(a.get());

	error::ErrptrT err = nullptr;
	auto refa = service.lookup_node(err, ida);
	ASSERT_EQ(a, refa);
	ASSERT_NOERR(err);
}


TEST(LOOKUP, RemoteLookupNode)
{
	size_t port1 = 5112;
	size_t port2 = 5113;

	const char* consul_addr = std::getenv("TEST_CONSUL_ADDRESS");
	if (nullptr == consul_addr || 0 == std::strlen(consul_addr))
	{
		consul_addr = "localhost";
	}
	std::string address = fmts::sprintf("http://%s:8500", consul_addr);
	auto consul = std::make_shared<ppconsul::Consul>(address);
	auto consulsvc = distr::make_consul(consul, port1, "Lookup.IOService", "service1");
	auto consulsvc2 = distr::make_consul(consul, port2, "Lookup.IOService", "service2");

	distr::PeerServiceConfig cfg(consulsvc, egrpc::ClientConfig(
		std::chrono::milliseconds(5000),
		std::chrono::milliseconds(10000),
		5
	));
	// use manager to serve service
	estd::ConfigMap<> mgrsvcs;
	distr::register_iosvc(mgrsvcs, cfg);
	distr::DistrManager manager(distr::ConsulSvcptrT(consulsvc), mgrsvcs, 1);

	distr::io::DistrIOService& service = distr::get_iosvc(manager);

	teq::Shape outshape({2, 2});
	auto a = std::make_shared<MockLeaf>(
		std::vector<double>{2, 3, 7, 2}, outshape);
	a->meta_.tcode_ = egen::DOUBLE;
	a->meta_.tname_ = "DOUBLE";
	service.expose_node(a);
	auto ida = *service.lookup_id(a.get());

	distr::PeerServiceConfig cfg2(consulsvc2, egrpc::ClientConfig(
		std::chrono::milliseconds(5000),
		std::chrono::milliseconds(10000),
		5
	));
	distr::io::DistrIOService service2(cfg2);

	error::ErrptrT err = nullptr;
	EXPECT_EQ(nullptr, service2.lookup_node(err, ida, false));
	ASSERT_NE(nullptr, err);
	auto expect_msg = fmts::sprintf(
		"no id %s found: will not recurse", ida.c_str());
	EXPECT_STREQ(expect_msg.c_str(), err->to_string().c_str());

	err = nullptr;
	auto refa = service2.lookup_node(err, ida);
	ASSERT_NE(nullptr, refa);
	ASSERT_NOERR(err);
	auto expect_refname = fmts::sprintf(
		"service1/%s", ida.c_str());
	EXPECT_STREQ(expect_refname.c_str(), refa->to_string().c_str());
}


#endif // DISABLE_IOSVC_LOOKUP_TEST
