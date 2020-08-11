#include "distrib/consul.hpp"

#ifdef DISTRIB_CONSUL_HPP

namespace distr
{

ConsulService* make_consul (
	ppconsul::Consul& consul, size_t port,
	const std::string& svc_name, const std::string& id)
{
	std::string svc_id = id.empty() ?
		boost::uuids::to_string(global::get_uuidengine()()) : id;
	return new ConsulService(
		consul, port, svc_id, svc_name);
}

}

#endif
