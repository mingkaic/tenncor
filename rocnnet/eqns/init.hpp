#include "ead/variable.hpp"
#include "ead/random.hpp"

#ifndef EQNS_INIT_HPP
#define EQNS_INIT_HPP

namespace eqns
{

template <typename T>
using InitF = std::function<ead::VarptrT<T>(ade::Shape,std::string)>;

template <typename T>
using ShapeFactorF = std::function<T(ade::Shape)>;

template <typename T>
T fanio (ade::Shape shape)
{
	return shape.at(0) + shape.at(1);
}

template <typename T>
T fanavg (ade::Shape shape)
{
	return fanio<T>(shape) / 2;
}

const size_t max_repick = 5;

template <typename T>
void truncated_normal (std::vector<T>& out, ade::Shape shape, T mean, T stdev)
{
	size_t n = shape.n_elems();
	out = std::vector<T>(n);
	auto gen = ead::norm_gen<T>(mean, stdev);
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

template <typename T>
InitF<T> variance_scaling_init (T factor, ShapeFactorF<T> sfactor=fanavg<T>)
{
	return
	[factor, sfactor](ade::Shape shape, std::string label)
	{
		std::vector<T> vec;
		T stdev = std::sqrt(factor / sfactor(shape));
		truncated_normal<T>(vec, shape, 0, stdev);
		return ead::make_variable(vec.data(), shape, label);
	};
}

template <typename T>
InitF<T> unif_xavier_init (T factor = 1)
{
	return
	[factor](ade::Shape shape, std::string label)
	{
		std::vector<T> vec(shape.n_elems());
		T bound = factor * std::sqrt(6.0 / fanio<T>(shape));
		std::generate(vec.begin(), vec.end(), ead::unif_gen<T>(-bound, bound));
		return ead::make_variable(vec.data(), shape, label);
	};
}

template <typename T>
InitF<T> norm_xavier_init (T factor = 1)
{
	return
	[factor](ade::Shape shape, std::string label)
	{
		std::vector<T> vec(shape.n_elems());
		T stdev = factor * std::sqrt(2.0 / fanio<T>(shape));
		std::generate(vec.begin(), vec.end(), ead::norm_gen<T>(0.0, stdev));
		return ead::make_variable(vec.data(), shape, label);
	};
}

}

#endif // EQNS_INIT_HPP