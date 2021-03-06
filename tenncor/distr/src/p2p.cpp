#include "tenncor/distr/p2p.hpp"

#ifdef DISTR_P2P_HPP

namespace distr
{

ConsulService* make_consul (ConsulptrT consul, size_t port,
	const std::string& svc_name, const std::string& id)
{
	std::string svc_id = id.empty() ? global::get_generator()->get_str() : id;
	return new ConsulService(
		consul, port, svc_id, svc_name);
}

}

#endif
