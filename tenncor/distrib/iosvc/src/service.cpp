
#include "tenncor/distrib/iosvc/service.hpp"

#ifdef DISTRIB_IO_SERVICE_HPP

namespace distr
{

error::ErrptrT register_iosvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	svcs.add_entry<io::DistrIOService>(io::iosvc_key,
		[&](){ return new io::DistrIOService(cfg); });
	return nullptr;
}

io::DistrIOService& get_iosvc (iDistrManager& manager)
{
	auto svc = manager.get_service(io::iosvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			io::iosvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<io::DistrIOService&>(*svc);
}

}

#endif
