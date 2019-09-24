#include "eteq/random.hpp"

#ifdef ETEQ_RANDOM_HPP

namespace eteq
{

EngineT& get_engine (void)
{
	static EngineT engine;
	return engine;
}

}

#endif
