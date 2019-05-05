#include <random>
#include <type_traits>

#ifndef EAD_RANDOM_HPP
#define EAD_RANDOM_HPP

namespace ead
{

/// RNG engine used
using EngineT = std::default_random_engine;

/// Return global random generator
EngineT& get_engine (void);

template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T unif (const T& a, const T& b)
{
	std::uniform_int_distribution<T> dist(a, b);
	return dist(get_engine());
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
T unif (const T& a, const T& b)
{
	std::uniform_real_distribution<T> dist(a, b);
	return dist(get_engine());
}

}

#endif // EAD_RANDOM_HPP
