#include <unordered_map>

#include "eteq/generated/pyapi.hpp"

#include "eteq/constant.hpp"
#include "eteq/variable.hpp"
#include "eteq/session.hpp"

#ifndef LAYR_ERR_APPROX_HPP
#define LAYR_ERR_APPROX_HPP

namespace layr
{

using VarErrsT = std::vector<std::pair<eteq::VarptrT<PybindT>,eteq::NodeptrT<PybindT>>>;

struct VarAssign
{
	std::string label_;

	eteq::VarptrT<PybindT> target_;

	eteq::NodeptrT<PybindT> source_;
};

using AssignsT = std::vector<VarAssign>;

using AssignGroupsT = std::vector<AssignsT>;

// approximate error of sources given error of root
using ApproxF = std::function<AssignGroupsT(const VarErrsT&)>;

using UpdateStepF = std::function<void(eteq::TensSetT&)>;

using NodeUnarF = std::function<eteq::NodeptrT<PybindT>(eteq::NodeptrT<PybindT>)>;

eteq::NodeptrT<PybindT> identity (eteq::NodeptrT<PybindT> node);

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

#endif // LAYR_ERR_APPROX_HPP
