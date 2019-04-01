#include <unordered_map>

#include "ead/generated/pyapi.hpp"

#include "ead/constant.hpp"
#include "ead/variable.hpp"
#include "ead/session.hpp"

#ifndef EQNS_ERR_APPROX_HPP
#define EQNS_ERR_APPROX_HPP

namespace eqns
{

using VariablesT = std::vector<ead::VarptrT<PybindT>>;

struct VarAssign
{
	std::string label_;

	ead::VarptrT<PybindT> target_;

	ead::NodeptrT<PybindT> source_;
};

using AssignsT = std::list<VarAssign>;

using AssignGroupsT = std::list<AssignsT>;

// approximate error of sources given error of root
using ApproxFuncT = std::function<AssignGroupsT(ead::NodeptrT<PybindT>&,VariablesT)>;

using UpdateStepT = std::function<void(std::unordered_set<ade::iTensor*>&)>;

// Stochastic Gradient Descent Approximation
AssignGroupsT sgd (ead::NodeptrT<PybindT>& root, VariablesT leaves,
	PybindT learning_rate = 0.5, std::string root_label = "");

// Momentum-based Root Mean Square Approximation
AssignGroupsT rms_momentum (ead::NodeptrT<PybindT>& root, VariablesT leaves,
	PybindT learning_rate = 0.5,
	PybindT discount_factor = 0.99,
	PybindT epsilon = std::numeric_limits<PybindT>::epsilon(),
	std::string root_label = "");

void assign_groups (AssignGroupsT& groups, UpdateStepT update_step);

}

#endif // EQNS_ERR_APPROX_HPP
