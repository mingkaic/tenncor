/// random.hpp
/// eigen
///
/// Purpose:
/// Define randomization functions used in Eigen operators
///

#include <random>
#include <type_traits>
#include <functional>

#ifndef EIGEN_RANDOM_HPP
#define EIGEN_RANDOM_HPP

namespace eigen
{

/// RNG engine used
using EngineT = std::default_random_engine;

/// Function that returns a generated number
template <typename T>
using GenF = std::function<T()>;

/// Return global random generator
EngineT& get_engine (void);

/// Return uniformly generated number between a and b (integers only)
template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T unif (const T& a, const T& b)
{
	std::uniform_int_distribution<T> dist(a, b);
	return dist(get_engine());
}

/// Return uniformly generate number between a and b (decimals only)
template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
T unif (const T& a, const T& b)
{
	std::uniform_real_distribution<T> dist(a, b);
	return dist(get_engine());
}

/// Return uniformly generator function that produces numbers between a and b (integers only)
template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
GenF<T> unif_gen (const T& a, const T& b)
{
	std::uniform_int_distribution<T> dist(a, b);
	return std::bind(dist, get_engine());
}

/// Return uniformly generator function that produces numbers between a and b (decimals only)
template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
GenF<T> unif_gen (T a, T b)
{
	std::uniform_real_distribution<T> dist(a, b);
	return std::bind(dist, get_engine());
}

/// Return normally generator function that produces numbers with mean and stdev (decimals only)
template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
GenF<T> norm_gen (T mean, T stdev)
{
	std::normal_distribution<T> dist(mean, stdev);
	return std::bind(dist, get_engine());
}

}

#endif // EIGEN_RANDOM_HPP
