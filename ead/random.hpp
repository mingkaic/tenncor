#include <random>
#include <type_traits>
#include <functional>

#ifndef EAD_RANDOM_HPP
#define EAD_RANDOM_HPP

namespace ead
{

/// RNG engine used
using EngineT = std::default_random_engine;

template <typename T>
using GenF = std::function<T()>;

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

template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
GenF<T> unif_gen (const T& a, const T& b)
{
	std::uniform_int_distribution<T> dist(a, b);
	return std::bind(dist, get_engine());
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
GenF<T> unif_gen (T a, T b)
{
	std::uniform_real_distribution<T> dist(a, b);
	return std::bind(dist, get_engine());
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
GenF<T> norm_gen (T mean, T stdev)
{
	std::normal_distribution<T> dist(mean, stdev);
	return std::bind(dist, get_engine());
}

}

#endif // EAD_RANDOM_HPP
