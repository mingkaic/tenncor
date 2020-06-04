#include "eigen/device.hpp"

#ifndef ETEQ_GLOBAL_HPP
#define ETEQ_GLOBAL_HPP

namespace eteq
{

using ETensRegistryT = std::unordered_map<void*,teq::TensptrT>;

struct ETensContext final
{
	eigen::iSessptrT sess_ = std::make_shared<teq::Session>();

	ETensRegistryT registry_;
};

using ECtxptrT = std::shared_ptr<ETensContext>;

using ECtxrefT = std::weak_ptr<ETensContext>;

ECtxptrT& global_context (void);

}

#endif // ETEQ_GLOBAL_HPP
