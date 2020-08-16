
#include "distrib/services/io/service.hpp"

#ifdef DISTRIB_IO_SERVICE_HPP

namespace distr
{

error::ErrptrT register_iosvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	svcs.add_entry<DistrIOService>(iosvc_key,
		[&](){ return new DistrIOService(cfg); });
	return nullptr;
}

DistrIOService& get_iosvc (iDistrManager& manager)
{
	auto svc = manager.get_service(iosvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			iosvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<DistrIOService&>(*svc);
}

}

#endif
