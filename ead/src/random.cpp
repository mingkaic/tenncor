#include "ead/random.hpp"

#ifdef EAD_RANDOM_HPP

namespace ead
{

EngineT& get_engine (void)
{
	static EngineT engine;
	return engine;
}

}

#endif
