/// random.hpp
/// eigen
///
/// Purpose:
/// Define randomization functions used in Eigen operators
///

#ifndef GLOBAL_RANDOM_HPP
#define GLOBAL_RANDOM_HPP

#include <random>
#include <type_traits>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "internal/global/config.hpp"

namespace global
{

/// RNG engine used
using RandEngineT = std::default_random_engine;

using BoostEngineT = boost::uuids::random_generator;

/// Function that returns a generated number
template <typename T>
using GenF = std::function<T()>;

void set_randengine (RandEngineT* reg,
	CfgMapptrT ctx = context());

RandEngineT& get_randengine (const CfgMapptrT& ctx = context());

void set_uuidengine (BoostEngineT* reg,
	CfgMapptrT ctx = context());

BoostEngineT& get_uuidengine (const CfgMapptrT& ctx = context());

void seed (size_t s, CfgMapptrT ctx = context());

struct Randomizer final
{
	Randomizer (const CfgMapptrT& ctx = context()) :
		engine_(&get_randengine(ctx)) {}

	/// Return uniformly generated number between a and b (integers only)
	template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
	T unif (const T& a, const T& b) const
	{
		std::uniform_int_distribution<T> dist(a, b);
		return dist(*engine_);
	}

	/// Return uniformly generate number between a and b (decimals only)
	template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
	T unif (const T& a, const T& b) const
	{
		std::uniform_real_distribution<T> dist(a, b);
		return dist(*engine_);
	}

	/// Return uniformly generator function that produces numbers between a and b (integers only)
	template <typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
	GenF<T> unif_gen (const T& a, const T& b) const
	{
		std::uniform_int_distribution<T> dist(a, b);
		return std::bind(dist, *engine_);
	}

	/// Return uniformly generator function that produces numbers between a and b (decimals only)
	template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
	GenF<T> unif_gen (T a, T b) const
	{
		std::uniform_real_distribution<T> dist(a, b);
		return std::bind(dist, *engine_);
	}

	/// Return normally generator function that produces numbers with mean and stdev (decimals only)
	template <typename T, typename std::enable_if<!std::is_integral<T>::value>::type* = nullptr>
	GenF<T> norm_gen (T mean, T stdev) const
	{
		std::normal_distribution<T> dist(mean, stdev);
		return std::bind(dist, *engine_);
	}

	mutable RandEngineT* engine_;
};

}

#endif // GLOBAL_RANDOM_HPP
