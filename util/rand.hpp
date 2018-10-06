///
/// rand.hpp
/// util
///
/// Purpose:
/// Define a single randomization engine and unique id function
///

#include <string>
#include <random>

#ifndef UTIL_RAND_HPP
#define UTIL_RAND_HPP

namespace util
{

/// RNG engine used
using EngineT = std::default_random_engine;

/// Return global random generator
EngineT& get_engine (void);

/// Return pseudo unique string given a pointer
/// The string aggregates the pointer, time, and random 16 character hex value
std::string make_uid (void* ptr, EngineT& engine = get_engine());

}

#endif // UTIL_RAND_HPP
