
#ifndef DISABLE_IOSVC_REMOTE_TEST


#include "gtest/gtest.h"

#include "exam/exam.hpp"

#include "internal/teq/mock/mock.hpp"

#include "tenncor/distr/distr.hpp"

#include "tenncor/distr/iosvc/iosvc.hpp"


TEST(REMOTE, LocalReferenceStorage)
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

	// shouldn't find references if we only exposed a from local
	EXPECT_EQ(0, service.get_remotes().size());

	auto ida = *service.lookup_id(a.get());
	error::ErrptrT err = nullptr;
	service.lookup_node(err, ida);

	// shouldn't find references
	EXPECT_EQ(0, service.get_remotes().size());
}


TEST(REMOTE, RemoteReferenceStorage)
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

	distr::PeerServiceConfig cfg2(consulsvc2, egrpc::ClientConfig(
		std::chrono::milliseconds(5000),
		std::chrono::milliseconds(10000),
		5
	));
	distr::io::DistrIOService service2(cfg2);

	EXPECT_EQ(0, service.get_remotes().size());

	auto ida = *service.lookup_id(a.get());
	error::ErrptrT err = nullptr;
	service2.lookup_node(err, ida);

	EXPECT_EQ(nullptr, err) << (nullptr == err ? "" : err->to_string());
	EXPECT_EQ(1, service2.get_remotes().size());
}


#endif // DISABLE_IOSVC_REMOTE_TEST
