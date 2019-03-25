#include "ead/generated/api.hpp"
#include "ead/generated/pyapi.hpp"

#include "ead/constant.hpp"
#include "ead/variable.hpp"

#ifndef EQNS_HELPER_HPP
#define EQNS_HELPER_HPP

namespace eqns
{

ead::NodeptrT<PybindT> one_binom (ead::NodeptrT<PybindT> a);

ead::NodeptrT<PybindT> weighed_bias_add (ead::NodeptrT<PybindT> weighed,
	ead::NodeptrT<PybindT> bias);

}

#endif // EQNS_HELPER_HPP
