#include <unordered_map>

#include "ead/constant.hpp"
#include "ead/variable.hpp"
#include "ead/session.hpp"

#ifndef EQNS_ERR_APPROX_HPP
#define EQNS_ERR_APPROX_HPP

namespace eqns
{

using VariablesT = std::vector<ead::VarptrT<double>>;

struct VarAssign
{
	ead::VarptrT<double> target_;

	ead::NodeptrT<double> source_;
};

using AssignsT = std::list<VarAssign>;

using AssignGroupsT = std::list<AssignsT>;

// approximate error of sources given error of root
using ApproxFuncT = std::function<AssignGroupsT(ead::NodeptrT<double>&,VariablesT)>;

// Stochastic Gradient Descent Approximation
AssignGroupsT sgd (ead::NodeptrT<double>& root, VariablesT leaves,
	double learning_rate = 0.5);

// Momentum-based Root Mean Square Approximation
AssignGroupsT rms_momentum (ead::NodeptrT<double>& root, VariablesT leaves,
	double learning_rate = 0.5,
	double discount_factor = 0.99,
	double epsilon = std::numeric_limits<double>::epsilon());

void assign_groups (ead::Session<double>& sess, AssignGroupsT& groups);

}

#endif // EQNS_ERR_APPROX_HPP
