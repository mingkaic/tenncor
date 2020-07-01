
#include "ppconsul/agent.h"
#include "ppconsul/catalog.h"

#ifndef DISTRIB_CONSUL_HPP
#define DISTRIB_CONSUL_HPP

namespace distrib
{

struct ConsulConfig final
{
	std::string id_;

	size_t port_;
};

struct ConsulService final
{
	ConsulService (ppconsul::Consul& consul, size_t port,
		const std::string& id, const std::string& service) :
		port_(port), id_(id), service_(service),
		agent_(consul), catalog_(consul)
	{
		teq::infof("serving %s:%s @ :%d",
			service_.c_str(), id_.c_str(), port_);
		agent_.registerService(
			ppconsul::agent::kw::name = service_,
			ppconsul::agent::kw::port = port_,
			ppconsul::agent::kw::id = id_,
			ppconsul::agent::kw::tags = {service_},
			ppconsul::agent::kw::check =
				ppconsul::agent::TtlCheck{std::chrono::seconds(5)}
		);

		agent_.servicePass(id_);
	}

	~ConsulService (void)
	{
		agent_.serviceFail(id_, "Shutting down");

		// Unregister service
		agent_.deregisterService(service_);
	}

	std::unordered_map<std::string,std::string> get_peers (void)
	{
		std::unordered_map<std::string,std::string> peers;
		auto services = catalog_.service(service_);
		for (auto& service : services)
		{
			auto id = service.second.id;
			if (id_ != id)
			{
				peers.emplace(id, fmts::sprintf("%s:%d",
					service.second.address.c_str(), service.second.port));
			}
		}
		return peers;
	}

	size_t port_;

	std::string id_;

	std::string service_;

	ppconsul::agent::Agent agent_;

	ppconsul::catalog::Catalog catalog_;
};

}

#endif // DISTRIB_CONSUL_HPP
