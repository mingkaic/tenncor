#include "eteq/global.hpp"

#ifdef ETEQ_GLOBAL_HPP

namespace eteq
{

ETensContext& global_context (void)
{
	static ETensContext registry;
	return registry;
}

}

#endif
