#include "internal/global/config.hpp"

#ifdef GLOBAL_CONFIG_HPP

namespace global
{

CfgMapptrT context (void)
{
	static CfgMapptrT ctx = std::make_shared<estd::ConfigMap<>>();
	return ctx;
}

}

#endif
