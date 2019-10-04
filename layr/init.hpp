///
/// init.hpp
/// layr
///
/// Purpose:
/// Define variable initialization algorithms
///

#include "eteq/variable.hpp"
#include "eteq/random.hpp"

#ifndef LAYR_INIT_HPP
#define LAYR_INIT_HPP

namespace layr
{

/// Function that produces a variable given the variable's shape and label
template <typename T>
using InitF = std::function<eteq::VarptrT<T>(teq::Shape,std::string)>;

/// Function that returns some metric of a shape
template <typename T>
using ShapeFactorF = std::function<T(teq::Shape)>;

/// Return the sum of the first 2 dimensions of a shape
template <typename T>
T fanio (teq::Shape shape)
{
	return shape.at(0) + shape.at(1);
}

/// Return the mean of the first 2 dimensions of a shape
template <typename T>
T fanavg (teq::Shape shape)
{
	return fanio<T>(shape) / 2;
}

/// Populate out vector with normally distributed values (using mean and stdev)
/// except repick values if the value is not within 2 stdev of the mean
template <typename T>
void truncated_normal (std::vector<T>& out, teq::Shape shape, T mean, T stdev,
	size_t max_repick = 5)
{
	size_t n = shape.n_elems();
	out = std::vector<T>(n);
	auto gen = eteq::norm_gen<T>(mean, stdev);
	std::generate(out.begin(), out.end(), gen);
	// if T is not decimal, program would fail to compile therefore T is signed
	T upperbound = mean + 2 * stdev;
	T lowerbound = mean - 2 * stdev;
	for (size_t i = 0; i < n; ++i)
	{
		// keep repicking until we give-up (statistical unlikely)
		for (size_t retry = 0;
			(out[i] > upperbound || out[i] < lowerbound) && max_repick;
			++retry)
		{
			out[i] = gen();
		}
		// clip
		if (out[i] > upperbound)
		{
			out[i] = upperbound;
		}
		else if (out[i] < lowerbound)
		{
			out[i] = lowerbound;
		}
	}
}

/// Return initialization function that makes zero variables
template <typename T>
InitF<T> zero_init (void)
{
	return
	[](teq::Shape shape, std::string label)
	{
		return eteq::make_variable_scalar<T>(0, shape, label);
	};
}

/// Return initialization function that makes variance scaling variables
/// (see https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/contrib/layers/variance_scaling_initializer)
template <typename T>
InitF<T> variance_scaling_init (T factor, ShapeFactorF<T> sfactor=fanavg<T>)
{
	return
	[factor, sfactor](teq::Shape shape, std::string label)
	{
		std::vector<T> vec;
		T stdev = std::sqrt(factor / sfactor(shape));
		truncated_normal<T>(vec, shape, 0, stdev);
		return eteq::make_variable(vec.data(), shape, label);
	};
}

/// Return initialization function that makes xavier initialized variables (that uses uniform distribution)
/// (see https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
template <typename T>
InitF<T> unif_xavier_init (T factor = 1)
{
	return
	[factor](teq::Shape shape, std::string label)
	{
		std::vector<T> vec(shape.n_elems());
		T bound = factor * std::sqrt(6. / fanio<T>(shape));
		std::generate(vec.begin(), vec.end(), eteq::unif_gen<T>(-bound, bound));
		return eteq::make_variable(vec.data(), shape, label);
	};
}

/// Return initialization function that makes xavier initialized variables (that uses gaussian distribution)
/// (see https://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
template <typename T>
InitF<T> norm_xavier_init (T factor = 1)
{
	return
	[factor](teq::Shape shape, std::string label)
	{
		std::vector<T> vec(shape.n_elems());
		T stdev = factor * std::sqrt(2. / fanio<T>(shape));
		std::generate(vec.begin(), vec.end(), eteq::norm_gen<T>(0., stdev));
		return eteq::make_variable(vec.data(), shape, label);
	};
}

}

#endif // LAYR_INIT_HPP
