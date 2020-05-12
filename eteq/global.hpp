#include "eigen/device.hpp"

#ifndef ETEQ_GLOBAL_HPP
#define ETEQ_GLOBAL_HPP

namespace eteq
{

using ETensRegistryT = std::unordered_map<void*,teq::TensptrT>;

struct ETensContext final
{
    teq::Session sess_ = eigen::get_session();

    ETensRegistryT registry_;
};

ETensContext& global_context (void);

}

#endif // ETEQ_GLOBAL_HPP
