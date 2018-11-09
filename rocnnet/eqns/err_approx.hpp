#include <unordered_map>

#include "adhoc/llo/node.hpp"

using DeltasT = std::unordered_map<llo::iSource*,llo::DataNode>;
// approximate error of sources given error of root
using ApproxFuncT = std::function<DeltasT(llo::DataNode&,
	std::vector<llo::DataNode>)>;

// Stochastic Gradient Descent Approximation
DeltasT sgd (llo::DataNode& root, std::vector<llo::DataNode> leaves,
	double learning_rate);

// Momentum-based Root Mean Square Approximation
DeltasT rms_momentum (llo::DataNode& root,
	std::vector<llo::DataNode> leaves, double learning_rate,
	double discount_factor, double epsilon);
