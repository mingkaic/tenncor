
#include "estd/config.hpp"

#ifndef GLOBAL_CONFIG_HPP
#define GLOBAL_CONFIG_HPP

namespace global
{

using CfgMapptrT = std::shared_ptr<estd::ConfigMap<>>;

using CfgMaprefT = std::weak_ptr<estd::ConfigMap<>>;

CfgMapptrT context (void);

}

#endif // GLOBAL_CONFIG_HPP
