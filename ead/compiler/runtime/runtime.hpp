#include <cmath>
#include <random>
#include <type_traits>

#include "Eigen/Core"

#include "ead/compiler/plugins/plugin.hpp"

#ifndef PLUGIN_RUNTIME_HPP
#define PLUGIN_RUNTIME_HPP

namespace compiler
{

/// RNG engine used
using EngineT = std::random_device;

/// Return global random generator
static EngineT& get_engine (void)
{
	static EngineT engine;
	return engine;
}

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

#endif // PLUGIN_RUNTIME_HPP
