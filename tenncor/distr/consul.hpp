
#ifndef DISTR_CONSUL_HPP
#define DISTR_CONSUL_HPP

#include "ppconsul/agent.h"
#include "ppconsul/catalog.h"
#include "ppconsul/kv.h"

#include "internal/global/global.hpp"

namespace distr
{

static const size_t consul_nretries = 5;

const std::string default_service = "tenncor";

using ConsulptrT = std::shared_ptr<ppconsul::Consul>;

struct ConsulService final
{
	ConsulService (ConsulptrT consul, size_t port,
		const std::string& id, const std::string& name) :
		consul_(consul), port_(port), id_(id), name_(name),
		agent_(*consul), catalog_(*consul), kv_(*consul)
	{
		global::infof("[consul %s] serving %s @ 0.0.0.0:%d",
			id_.c_str(), name_.c_str(), port_);
		agent_.registerService(
			ppconsul::agent::kw::name = name_,
			ppconsul::agent::kw::port = port_,
			ppconsul::agent::kw::id = id_,
			ppconsul::agent::kw::tags = {name_},
			ppconsul::agent::kw::check =
				ppconsul::agent::TtlCheck{std::chrono::seconds(5)}
		);

		agent_.servicePass(id_);
	}

	~ConsulService (void)
	{
		agent_.serviceFail(id_, "Shutting down");

		// Unregister service
		agent_.deregisterService(id_);
	}

	types::StrUMapT<std::string> get_peers (void)
	{
		types::StrUMapT<std::string> peers;
		std::vector<ppconsul::catalog::NodeService> services;
		for (size_t i = 0; i < consul_nretries; ++i)
		{
			try
			{
				services = catalog_.service(name_);
				break;
			}
			catch (...)
			{
				global::warnf("consul failed to get service, retry %d", i);
			}
		}
		for (auto& service : services)
		{
			auto id = service.second.id;
			if (id_ != id)
			{
				std::string address = service.second.address;
				if (address.empty())
				{
					address = "localhost";
				}
				peers.emplace(id, fmts::sprintf("%s:%d",
					address.c_str(), service.second.port));
			}
		}
		return peers;
	}

	void set_kv (const std::string& key, const std::string& value)
	{
		kv_.set(key, value);
	}

	std::string get_kv (const std::string& key,
		const std::string& default_val)
	{
		return kv_.get(key, default_val, ppconsul::kv::kw::consistency =
			ppconsul::Consistency::Consistent);
	}

	ConsulptrT consul_;

	size_t port_;

	std::string id_;

	std::string name_;

	ppconsul::agent::Agent agent_;

	ppconsul::catalog::Catalog catalog_;

	ppconsul::kv::Kv kv_;
};

ConsulService* make_consul (ConsulptrT consul, size_t port,
	const std::string& svc_name = default_service,
	const std::string& id = "");

}

#endif // DISTR_CONSUL_HPP
