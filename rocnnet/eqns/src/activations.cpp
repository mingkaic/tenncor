#include "rocnnet/eqns/activations.hpp"

#ifdef EQNS_ACTIVATIONS_HPP

namespace eqns
{

ead::NodeptrT<PybindT> identity (ead::NodeptrT<PybindT> x)
{
	return x;
}

ead::NodeptrT<PybindT> softmax (ead::NodeptrT<PybindT> x)
{
	auto num = age::exp(x);
	auto denom = age::reduce_sum(num);
	ade::Shape shape = x->shape();
	return age::div(num, age::extend(denom, 0,
		std::vector<uint8_t>(shape.begin(), shape.end())));
}

}

#endif
