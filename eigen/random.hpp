/// random.hpp
/// eigen
///
/// Purpose:
/// Define randomization functions used in Eigen operators
///

#ifndef EIGEN_RANDOM_HPP
#define EIGEN_RANDOM_HPP

#include <random>
#include <type_traits>

#include "teq/teq.hpp"

namespace eigen
{

const std::string rengine_key = "rengine";

/// RNG engine used
using EngineT = std::default_random_engine;

/// Function that returns a generated number
template <typename T>
using GenF = std::function<T()>;

/// Return global random generator
EngineT& default_engine (void);

struct Randomizer final
{
	Randomizer (void) : engine_(static_cast<EngineT*>(
		config::global_config.get_obj(rengine_key)))
	{
		// fallback to default engine
		if (nullptr == engine_)
		{
			teq::error("missing random engine in global config");
			engine_ = &default_engine();
		}
	}

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

	mutable EngineT* engine_;
};

#define RANDOM_INIT ::config::global_config.add_entry<eigen::EngineT>(eigen::rengine_key)

}

#endif // EIGEN_RANDOM_HPP
