#include "eigen/random.hpp"

#ifdef EIGEN_RANDOM_HPP

namespace eigen
{

boost::uuids::random_generator& rand_uuid_gen (void)
{
	static boost::uuids::random_generator gen;
	return gen;
}

EngineT& default_engine (void)
{
	static EngineT engine;
	return engine;
}

}

#endif
