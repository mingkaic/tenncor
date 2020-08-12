
#include "dbg/distr_ext/print/service.hpp"

#ifdef DISTRIB_PRINT_SERVICE_HPP

namespace distr
{

error::ErrptrT register_printsvc (estd::ConfigMap<>& svcs,
	const PeerServiceConfig& cfg)
{
	auto iosvc = static_cast<DistrIOService*>(svcs.get_obj(iosvc_key));
	if (nullptr == iosvc)
	{
		return error::error("printsvc requires iosvc already registered");
	}
	svcs.add_entry<DistrPrintService>(printsvc_key,
		[&](){ return new DistrPrintService(cfg, iosvc); });
	return nullptr;
}

DistrPrintService& get_printsvc (iDistrManager& manager)
{
	auto svc = manager.get_service(printsvc_key);
	if (nullptr == svc)
	{
		global::fatalf("%s service not found in %s",
			printsvc_key.c_str(), manager.get_id().c_str());
	}
	return static_cast<DistrPrintService&>(*svc);
}

}

#endif
