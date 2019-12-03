///
/// err_approx.hpp
/// layr
///
/// Purpose:
/// Define error approximation algorithms and variable assignment utilities
///

#include <unordered_map>

#include "teq/session.hpp"

#include "eteq/generated/pyapi.hpp"
#include "eteq/constant.hpp"
#include "eteq/variable.hpp"

#ifndef LAYR_ERR_APPROX_HPP
#define LAYR_ERR_APPROX_HPP

namespace layr
{

/// Ordered association between variable and error
using VarErrsT = std::vector<std::pair<eteq::VarptrT<PybindT>,NodeptrT>>;

/// Variable and error approximation assignment encapsulation
struct VarAssign
{
	/// Representation of assignment
	std::string label_;

	/// Variable to assign to
	eteq::VarptrT<PybindT> target_;

	/// Variable update as to minimize the error in future iterations
	NodeptrT source_;
};

/// One batch of assignments
using AssignsT = std::vector<VarAssign>;

/// All batches of assignments
using AssignGroupsT = std::vector<AssignsT>;

/// Function that returns the error between two nodes,
/// left node contains expected values, right contains resulting values
using ErrorF = std::function<NodeptrT(NodeptrT,NodeptrT)>;

/// Function that approximate error of sources
/// given a vector of variables and its corresponding errors
using ApproxF = std::function<AssignGroupsT(const VarErrsT&)>;

/// Function that runs before or after variable assignment
/// to calculate approximation graphs
using UpdateStepF = std::function<void(teq::TensSetT&)>;

/// Return square(expect - got)
NodeptrT sqr_diff (NodeptrT expect, NodeptrT got);

/// Return all batches of variable assignments of
/// stochastic gradient descent error approximation applied to
/// particular variables-error associations
///
/// Stochastic Gradient Descent Approximation
/// for each (x, err) in leaves
/// x_next ~ x_curr - η * err,
///
/// where η is the learning rate
AssignGroupsT sgd (const VarErrsT& leaves,
	PybindT learning_rate = 0.5, std::string root_label = "");

AssignGroupsT adagrad (const VarErrsT& leaves, PybindT learning_rate,
	PybindT epsilon, std::string root_label = "");

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
AssignGroupsT rms_momentum (const VarErrsT& leaves,
	PybindT learning_rate = 0.5, PybindT discount_factor = 0.99,
	PybindT epsilon = std::numeric_limits<PybindT>::epsilon(),
	std::string root_label = "");

/// Apply all batches of assignments with update_step applied after each batch
void assign_groups (const AssignGroupsT& groups, UpdateStepF update_step);

/// Apply all batches of assignments with update_step applied before each batch
void assign_groups_preupdate (const AssignGroupsT& groups, UpdateStepF update_step);

}

#endif // LAYR_ERR_APPROX_HPP
