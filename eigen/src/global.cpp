#include "eigen/global.hpp"

#ifdef EIGEN_GLOBAL_HPP

namespace eigen
{

CtxptrT& global_context (void)
{
	static CtxptrT registry = std::make_shared<TensContext>();
	return registry;
}

}

#endif
