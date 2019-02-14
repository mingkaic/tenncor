#include "ead/generated/api.hpp"

#include "ead/constant.hpp"

#ifndef EQNS_ACTIVATIONS_HPP
#define EQNS_ACTIVATIONS_HPP

namespace eqns
{

/// sigmoid function: f(x) = 1/(1+e^-x)
ead::NodeptrT<double> sigmoid (ead::NodeptrT<double> x);

ead::NodeptrT<double> slow_sigmoid (ead::NodeptrT<double> x);

/// tanh function: f(x) = (e^(2*x)+1)/(e^(2*x)-1)
ead::NodeptrT<double> tanh (ead::NodeptrT<double> x);

/// softmax function: f(x) = e^x / sum(e^x)
ead::NodeptrT<double> softmax (ead::NodeptrT<double> x);

}

#endif
