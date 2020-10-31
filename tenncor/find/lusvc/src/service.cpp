
#include "tenncor/find/lusvc/service.hpp"

#ifdef DISTR_LU_SERVICE_HPP

namespace distr
{

error::ErrptrT register_lusvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<io::DistrIOService*>(svcs.get_obj(io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("opsvc requires iosvc already registered");
	}
	svcs.add_entry<lu::DistrLuService>(lu::lusvc_key,
		[&]{ return new lu::DistrLuService(cfg, iosvc); });
	return nullptr;
}

lu::DistrLuService& get_lusvc (iDistrManager& manager)
{
	auto svc = manager.get_service(lu::lusvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			lu::lusvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<lu::DistrLuService&>(*svc);
}

}

#endif
