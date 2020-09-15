
#ifndef DISTR_MOCK_MAKE_MGR_HPP
#define DISTR_MOCK_MAKE_MGR_HPP

#include "tenncor/distr/distr.hpp"

struct DistrTestcase
{
	DistrTestcase (const std::string service_name) :
		service_name_(service_name)
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

	virtual ~DistrTestcase (void)
	{
		clean_up();
	}

protected:
	distr::iDistrMgrptrT make_mgr (size_t port,
		const std::vector<distr::RegisterSvcF>& services,
		const std::string& alias = "")
	{
		auto consulsvc = distr::make_consul(consul_, port, service_name_, alias);
		distr::PeerServiceConfig cfg(consulsvc, egrpc::ClientConfig(
				std::chrono::milliseconds(5000),
				std::chrono::milliseconds(10000),
				5
			));
		estd::ConfigMap<> svcs;
		for (auto& reg : services)
		{
			assert(nullptr == reg(svcs, cfg));
		}
		return std::make_shared<distr::DistrManager>(
			distr::ConsulSvcptrT(consulsvc), svcs);
	}

	void clean_up (void)
	{
		ppconsul::agent::Agent agent(*consul_);
		ppconsul::catalog::Catalog catalog(*consul_);
		auto services = catalog.service(service_name_);
		for (auto& service : services)
		{
			agent.deregisterService(service.second.id);
		}
	}

	distr::ConsulptrT consul_;

	std::string service_name_;
};

#endif // DISTR_MOCK_MAKE_MGR_HPP
