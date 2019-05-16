#include "ead/generated/api.hpp"
#include "ead/generated/pyapi.hpp"

#include "ead/constant.hpp"

#ifndef EQNS_ACTIVATIONS_HPP // deprecated by prx
#define EQNS_ACTIVATIONS_HPP

namespace eqns
{

/// do nothing
ead::NodeptrT<PybindT> identity (ead::NodeptrT<PybindT> x);

/// softmax function: f(x) = e^x / sum(e^x)
ead::NodeptrT<PybindT> softmax (ead::NodeptrT<PybindT> x);

}

#endif
