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
	ead::VarptrT<PybindT> target_;

	ead::NodeptrT<PybindT> source_;
};

using AssignsT = std::list<VarAssign>;

using AssignGroupsT = std::list<AssignsT>;

// approximate error of sources given error of root
using ApproxFuncT = std::function<AssignGroupsT(ead::NodeptrT<PybindT>&,VariablesT)>;

// Stochastic Gradient Descent Approximation
AssignGroupsT sgd (ead::NodeptrT<PybindT>& root, VariablesT leaves,
	PybindT learning_rate = 0.5);

// Momentum-based Root Mean Square Approximation
AssignGroupsT rms_momentum (ead::NodeptrT<PybindT>& root, VariablesT leaves,
	PybindT learning_rate = 0.5,
	PybindT discount_factor = 0.99,
	PybindT epsilon = std::numeric_limits<PybindT>::epsilon());

void assign_groups (ead::Session<PybindT>& sess, AssignGroupsT& groups);

}

#endif // EQNS_ERR_APPROX_HPP
