///
/// init.hpp
/// layr
///
/// Purpose:
/// Define variable initialization algorithms
///

#ifndef LAYR_INIT_HPP
#define LAYR_INIT_HPP

#include "tenncor/eteq/eteq.hpp"

namespace layr
{

/// Function that produces a variable given the variable's shape and label
template <typename T>
using InitF = std::function<eteq::EVariable<T>(teq::Shape,std::string)>;

/// Function that returns some metric of a shape
template <typename T>
using ShapeFactorF = std::function<T(teq::Shape)>;

/// Return the sum of the first 2 dimensions of a shape
template <typename T>
T fanio (teq::Shape shape)
{
	auto slist = teq::narrow_shape(shape);
	return std::accumulate(slist.begin(), slist.end(), (teq::DimT) 0);
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
	auto gen = global::get_generator()->norm_decgen(mean, stdev);
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

}

#endif // LAYR_INIT_HPP
