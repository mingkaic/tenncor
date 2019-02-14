#include <unordered_map>

#include "ead/variable.hpp"
#include "ead/session.hpp"

#ifndef EQNS_ERR_APPROX_HPP
#define EQNS_ERR_APPROX_HPP

namespace eqns
{

using AssignFuncT = std::function<void(ead::Session<double>&)>;

using VariablesT = std::vector<ead::VarptrT<double>>;

struct Deltas
{
	void assign (ead::Session<double>& sess)
	{
		for (auto& action : actions_)
		{
			action(sess);
		}
	}

	// steps to update
	std::vector<AssignFuncT> actions_;

	// nodes upkept by approximation process
	std::vector<ead::NodeptrT<double>> upkeep_;
};

// approximate error of sources given error of root
using ApproxFuncT = std::function<Deltas(ead::NodeptrT<double>&,VariablesT)>;

// todo: change from map to vector of pairs, since we never make use of varmap's constant-time access
using VarmapT = std::unordered_map<ead::VarptrT<double>,ead::NodeptrT<double>>;

inline void assign_all (ead::Session<double>& sess, VarmapT connection);

// Stochastic Gradient Descent Approximation
Deltas sgd (ead::NodeptrT<double>& root, VariablesT leaves,
	double learning_rate = 0.5);

// Momentum-based Root Mean Square Approximation
Deltas rms_momentum (ead::NodeptrT<double>& root, VariablesT leaves,
	double learning_rate = 0.5,
	double discount_factor = 0.99,
	double epsilon = std::numeric_limits<double>::epsilon());

}

#endif // EQNS_ERR_APPROX_HPP
