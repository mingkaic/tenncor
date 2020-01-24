///
/// approx.hpp
/// layr
///
/// Purpose:
/// Define error approximation algorithms and variable assignment utilities
///

#include "eteq/generated/api.hpp"
#include "eteq/generated/pyapi.hpp"
#include "eteq/make.hpp"

#ifndef LAYR_APPROX_HPP
#define LAYR_APPROX_HPP

namespace layr
{

template <typename T>
using VarErrT = std::pair<eteq::EVariable<T>,eteq::ETensor<T>>;

/// Ordered association between variable and error
template <typename T>
using VarErrsT = std::vector<VarErrT<T>>;

/// Function that returns the error between two nodes,
/// left node contains expected values, right contains resulting values
template <typename T>
using ErrorF = std::function<eteq::ETensor<T>(eteq::ETensor<T>,eteq::ETensor<T>)>;

/// Function that approximate error of sources
/// given a vector of variables and its corresponding errors
template <typename T>
using ApproxF = std::function<eteq::ETensorsT<T>(const VarErrsT<T>&)>;

/// Function that runs before or after variable assignment
/// to calculate approximation graphs
using UpdateStepF = std::function<void(teq::TensSetT&)>;

/// Return square(expect - got)
template <typename T>
eteq::ETensor<T> sqr_diff (eteq::ETensor<T> expect, eteq::ETensor<T> got)
{
	return tenncor::square(expect - got);
}

/// Return all batches of variable assignments of
/// stochastic gradient descent error approximation applied to
/// particular variables-error associations
///
/// Stochastic Gradient Descent Approximation
/// for each (x, err) in leaves
/// x_next ~ x_curr - η * err,
///
/// where η is the learning rate
template <typename T>
eteq::ETensorsT<T> sgd (const VarErrsT<T>& assocs, T learning_rate = 0.5)
{
	eteq::ETensorsT<T> assigns;
	assigns.reserve(assocs.size());
	std::transform(assocs.begin(), assocs.end(),
	std::back_inserter(assigns),
	[&](VarErrT<T> verrs)
	{
		return tenncor::assign_sub(
			verrs.first, verrs.second * learning_rate);
	});
	return assigns;
}

template <typename T>
eteq::ETensorsT<T> adagrad (const VarErrsT<T>& assocs, T learning_rate = 0.5,
	T epsilon = std::numeric_limits<T>::epsilon())
{
	eteq::ETensorsT<T> assigns;
	assigns.reserve(assocs.size());
	std::transform(assocs.begin(), assocs.end(),
	std::back_inserter(assigns),
	[&](VarErrT<T> verrs)
	{
		auto& grad = verrs.second;
		eteq::EVariable<T> momentum = eteq::make_variable_like<T>(
			1, grad, "momentum");
		auto umom = tenncor::assign_add(momentum, tenncor::square(grad));

		// assign momentums before leaves
		return tenncor::assign_sub(
			verrs.first, grad * learning_rate /
			(tenncor::sqrt(umom) + epsilon));
	});
	return assigns;
}

/// Return all batches of variable assignments of
/// momentum-based rms error approximation applied to
/// particular variables-error associations
///
/// Momentum-based Root Mean Square Approximation
/// for each (x, err) in leaves
/// momentum_next ~ χ * momentum_cur + (1 - χ) * err ^ 2
/// x_next ~ x_curr - (η * err) / (sqrt(ε + momentum_next))
///
/// where η is the learning rate, ε is epsilon,
/// and χ is discount_factor
/// initial momentum is 1
template <typename T>
eteq::ETensorsT<T> rms_momentum (const VarErrsT<T>& assocs,
	T learning_rate = 0.5, T discount_factor = 0.99,
	T epsilon = std::numeric_limits<T>::epsilon())
{
	eteq::ETensorsT<T> assigns;
	assigns.reserve(assocs.size());
	std::transform(assocs.begin(), assocs.end(),
	std::back_inserter(assigns),
	[&](VarErrT<T> verrs)
	{
		auto& grad = verrs.second;
		eteq::EVariable<T> momentum = eteq::make_variable_like<T>(
			1, grad, "momentum");
		auto umom = tenncor::assign(momentum, discount_factor * momentum +
			(T(1) - discount_factor) * tenncor::square(grad));

		// assign momentums before leaves
		return tenncor::assign_sub(
			verrs.first, grad * learning_rate /
			(tenncor::sqrt(umom) + epsilon));
	});
	return assigns;
}

}

#endif // LAYR_APPROX_HPP
