#include "ead/generated/api.hpp"
#include "ead/constant.hpp"
#include "ead/variable.hpp"

#ifndef EQNS_HELPER_HPP
#define EQNS_HELPER_HPP

namespace eqns
{

ead::NodeptrT<double> one_binom (ead::NodeptrT<double> a);

ead::NodeptrT<double> weighed_bias_add (ead::NodeptrT<double> weighed,
	ead::NodeptrT<double> bias);

}

#endif // EQNS_HELPER_HPP
