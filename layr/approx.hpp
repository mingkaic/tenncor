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

/// Ordered association between variable and error
template <typename T>
using VarMapT = std::unordered_map<eteq::VarptrT<T>,eteq::ETensor<T>>;

/// Function that returns the error between two nodes,
/// left node contains expected values, right contains resulting values
template <typename T>
using ErrorF = std::function<eteq::ETensor<T>(eteq::ETensor<T>,eteq::ETensor<T>)>;

/// Function that approximate error of sources
/// given a vector of variables and its corresponding errors
template <typename T>
using ApproxF = std::function<VarMapT<T>(const VarMapT<T>&)>;

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
VarMapT<T> sgd (const VarMapT<T>& assocs, T learning_rate = 0.5)
{
	VarMapT<T> out;
	for (const auto& assoc : assocs)
	{
		out.emplace(assoc.first, tenncor::assign_sub(
			eteq::EVariable<T>(assoc.first), assoc.second * learning_rate));
	}
	return out;
}

template <typename T>
VarMapT<T> adagrad (const VarMapT<T>& assocs, T learning_rate = 0.5,
	T epsilon = std::numeric_limits<T>::epsilon())
{
	VarMapT<T> out;
	for (const auto& assoc : assocs)
	{
		eteq::EVariable<T> momentum = eteq::make_variable_like<T>(
			1, assoc.second, "momentum");
		auto update = tenncor::assign_add(momentum,
			tenncor::square(assoc.second));

		// assign momentums before leaves
		out.emplace(assoc.first, tenncor::assign_sub(
			eteq::EVariable<T>(assoc.first), assoc.second * learning_rate /
			(tenncor::sqrt(update) + epsilon)));
	}
	return out;
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
VarMapT<T> rms_momentum (const VarMapT<T>& assocs,
	T learning_rate = 0.5, T discount_factor = 0.99,
	T epsilon = std::numeric_limits<T>::epsilon())
{
	VarMapT<T> out;
	for (const auto& assoc : assocs)
	{
		eteq::EVariable<T> momentum = eteq::make_variable_like<T>(
			1, assoc.second, "momentum");
		auto update = tenncor::assign(momentum, discount_factor * momentum +
			(T(1) - discount_factor) * tenncor::square(assoc.second));

		// assign momentums before leaves
		out.emplace(assoc.first, tenncor::assign_sub(
			eteq::EVariable<T>(assoc.first), assoc.second * learning_rate /
			(tenncor::sqrt(update) + epsilon)));
	}
	return out;
}

}

#endif // LAYR_APPROX_HPP
