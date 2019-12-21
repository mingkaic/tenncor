///
/// err_approx.hpp
/// layr
///
/// Purpose:
/// Define error approximation algorithms and variable assignment utilities
///

#include <unordered_map>

#include "teq/session.hpp"

#include "eteq/generated/api.hpp"
#include "eteq/generated/pyapi.hpp"
#include "eteq/make.hpp"

#ifndef LAYR_ERR_APPROX_HPP
#define LAYR_ERR_APPROX_HPP

namespace layr
{

template <typename T>
using VarErrT = std::pair<eteq::VarptrT<T>,eteq::ETensor<T>>;

/// Ordered association between variable and error
template <typename T>
using VarErrsT = std::vector<VarErrT<T>>;

/// Variable and error approximation assignment encapsulation
template <typename T>
struct VarAssign
{
	/// Variable to assign to
	eteq::VarptrT<T> target_;

	/// Variable update as to minimize the error in future iterations
	eteq::ETensor<T> source_;
};

/// One batch of assignments
template <typename T>
using AssignsT = std::vector<VarAssign<T>>;

/// All batches of assignments
template <typename T>
using AssignGroupsT = std::vector<AssignsT<T>>;

/// Function that returns the error between two nodes,
/// left node contains expected values, right contains resulting values
template <typename T>
using ErrorF = std::function<eteq::ETensor<T>(eteq::ETensor<T>,eteq::ETensor<T>)>;

/// Function that approximate error of sources
/// given a vector of variables and its corresponding errors
template <typename T>
using ApproxF = std::function<AssignGroupsT<T>(const VarErrsT<T>&)>;

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
AssignGroupsT<T> sgd (const VarErrsT<T>& assocs,
	T learning_rate = 0.5)
{
	AssignsT<T> assignments;
	assignments.reserve(assocs.size());
	std::transform(assocs.begin(), assocs.end(),
	std::back_inserter(assignments),
	[&](VarErrT<T> verrs)
	{
		return VarAssign<T>{verrs.first, eteq::ETensor<T>(verrs.first) -
			verrs.second * learning_rate};
	});
	return {assignments};
}

template <typename T>
AssignGroupsT<T> adagrad (const VarErrsT<T>& assocs,
	T learning_rate = 0.5,
	T epsilon = std::numeric_limits<T>::epsilon())
{
	// assign momentums before leaves
	size_t nassocs = assocs.size();
	AssignsT<T> momentums;
	AssignsT<T> leaves;
	momentums.reserve(nassocs);
	leaves.reserve(nassocs);
	for (size_t i = 0; i < nassocs; ++i)
	{
		eteq::VarptrT<T> momentum = eteq::make_variable_like<T>(
			1, assocs[i].second, "momentum");

		auto next_momentum = eteq::ETensor<T>(momentum) +
			tenncor::square(assocs[i].second);
		auto leaf_next = eteq::ETensor<T>(assocs[i].first) -
			assocs[i].second * learning_rate /
			(tenncor::sqrt(eteq::ETensor<T>(momentum)) + epsilon);
		momentums.push_back(VarAssign<T>{momentum, next_momentum});
		leaves.push_back(VarAssign<T>{assocs[i].first, leaf_next});
	}
	return {momentums, leaves};
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
AssignGroupsT<T> rms_momentum (const VarErrsT<T>& assocs,
	T learning_rate = 0.5,
	T discount_factor = 0.99,
	T epsilon = std::numeric_limits<T>::epsilon())
{
	// assign momentums before leaves
	size_t nassocs = assocs.size();
	AssignsT<T> momentums;
	AssignsT<T> leaves;
	momentums.reserve(nassocs);
	leaves.reserve(nassocs);
	for (size_t i = 0; i < nassocs; ++i)
	{
		eteq::VarptrT<T> momentum = eteq::make_variable_like<T>(
			1, assocs[i].second, "momentum");

		auto momentum_next = discount_factor * eteq::ETensor<T>(momentum) +
			((T) 1 - discount_factor) * tenncor::square(assocs[i].second);
		auto leaf_next = eteq::ETensor<T>(assocs[i].first) -
			assocs[i].second * learning_rate /
			(tenncor::sqrt(eteq::ETensor<T>(momentum)) + epsilon);
		momentums.push_back(VarAssign<T>{momentum, momentum_next});
		leaves.push_back(VarAssign<T>{assocs[i].first, leaf_next});
	}
	return {momentums, leaves};
}

/// Apply all batches of assignments with update_step applied after each batch
template <typename T>
void assign_groups (
	const AssignGroupsT<T>& groups, UpdateStepF update_step)
{
	for (const AssignsT<T>& group : groups)
	{
		teq::TensSetT updated_var;
		for (const VarAssign<T>& assign : group)
		{
			updated_var.emplace(assign.target_.get());
			assign.target_->assign(*assign.source_);
		}
		update_step(updated_var);
	}
}

/// Apply all batches of assignments with update_step applied before each batch
template <typename T>
void assign_groups_preupdate (
	const AssignGroupsT<T>& groups, UpdateStepF update_step)
{
	for (const AssignsT<T>& group : groups)
	{
		teq::TensSetT sources;
		for (const VarAssign<T>& assign : group)
		{
			sources.emplace(assign.source_.get());
		}
		update_step(sources);
		for (const VarAssign<T>& assign : group)
		{
			assign.target_->assign(*assign.source_);
		}
	}
}

}

#endif // LAYR_ERR_APPROX_HPP
