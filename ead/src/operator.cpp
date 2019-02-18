#include "ead/operator.hpp"

#ifdef EAD_OPERATOR_HPP

namespace ead
{

/// Return global random generator
EngineT& get_engine (void)
{
	static EngineT engine;
	return engine;
}

}

#endif
