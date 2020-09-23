
#include "tenncor/serial/oxsvc/service.hpp"

#ifdef DISTR_OX_SERVICE_HPP

namespace distr
{

error::ErrptrT register_oxsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<io::DistrIOService*>(svcs.get_obj(io::iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("opsvc requires iosvc already registered");
	}
	svcs.add_entry<ox::DistrSerializeService>(ox::oxsvc_key,
		[&](){ return new ox::DistrSerializeService(cfg, iosvc); });
	return nullptr;
}

ox::DistrSerializeService& get_oxsvc (iDistrManager& manager)
{
	auto svc = manager.get_service(ox::oxsvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			ox::oxsvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<ox::DistrSerializeService&>(*svc);
}

}

#endif
