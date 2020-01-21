#include "eigen/random.hpp"

#ifdef EIGEN_RANDOM_HPP

namespace eigen
{

EngineT& default_engine (void)
{
	static EngineT engine;
	return engine;
}

}

#endif
