#include "tenncor/hone/hosvc/service.hpp"

#ifdef DISTRIB_HO_SERVICE_HPP

namespace distr
{

error::ErrptrT register_hosvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<io::DistrIOService*>(svcs.get_obj(io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("hosvc requires iosvc already registered");
	}
	svcs.add_entry<ho::DistrHoService>(ho::hosvc_key,
		[&](){ return new ho::DistrHoService(cfg, iosvc); });
	return nullptr;
}

ho::DistrHoService& get_hosvc (iDistrManager& manager)
{
	auto svc = manager.get_service(ho::hosvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			ho::hosvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<ho::DistrHoService&>(*svc);
}

}

#endif
