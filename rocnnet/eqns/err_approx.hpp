#include <unordered_map>

#include "ead/generated/pyapi.hpp"

#include "ead/constant.hpp"
#include "ead/variable.hpp"
#include "ead/session.hpp"

#ifndef EQNS_ERR_APPROX_HPP
#define EQNS_ERR_APPROX_HPP

namespace eqns
{

using VarErrsT = std::vector<std::pair<ead::VarptrT<PybindT>,ead::NodeptrT<PybindT>>>;

struct VarAssign
{
	std::string label_;

	ead::VarptrT<PybindT> target_;

	ead::NodeptrT<PybindT> source_;
};

using AssignsT = std::list<VarAssign>;

using AssignGroupsT = std::list<AssignsT>;

// approximate error of sources given error of root
using ApproxF = std::function<AssignGroupsT(const VarErrsT&)>;

using UpdateStepF = std::function<void(ead::TensSetT&)>;

using NodeUnarF = std::function<ead::NodeptrT<PybindT>(ead::NodeptrT<PybindT>)>;

ead::NodeptrT<PybindT> identity (ead::NodeptrT<PybindT> node);

// Stochastic Gradient Descent Approximation
// for each (x, err) in leaves
// x_next ~ x_curr - η * err,
//
// where η is the learning rate
AssignGroupsT sgd (const VarErrsT& leaves,
	PybindT learning_rate = 0.5, std::string root_label = "");

// Momentum-based Root Mean Square Approximation
// for each (x, err) in leaves
// momentum_next ~ χ * momentum_cur + (1 - χ) * err ^ 2
// x_next ~ x_curr - (η * err) / (sqrt(ε + momentum_next))
//
// where η is the learning rate, ε is epsilon,
// and χ is discount_factor
AssignGroupsT rms_momentum (const VarErrsT& leaves,
	PybindT learning_rate = 0.5, PybindT discount_factor = 0.99,
	PybindT epsilon = std::numeric_limits<PybindT>::epsilon(),
	std::string root_label = "");

void assign_groups (AssignGroupsT& groups, UpdateStepF update_step);

}

#endif // EQNS_ERR_APPROX_HPP
