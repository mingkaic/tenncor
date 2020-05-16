#include "eteq/global.hpp"

#ifdef ETEQ_GLOBAL_HPP

namespace eteq
{

ECtxptrT& global_context (void)
{
	static ECtxptrT registry = std::make_shared<ETensContext>();
	return registry;
}

}

#endif
