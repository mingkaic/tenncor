
#include "dbg/distr_ext/print/service.hpp"

#ifdef DISTRIB_PRINT_SERVICE_HPP

namespace distr
{

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
