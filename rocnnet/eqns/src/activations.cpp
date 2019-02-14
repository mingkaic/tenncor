#include "rocnnet/eqns/activations.hpp"

#ifdef EQNS_ACTIVATIONS_HPP

namespace eqns
{

ead::NodeptrT<double> sigmoid (ead::NodeptrT<double> x)
{
	return age::sigmoid(x);
}

ead::NodeptrT<double> tanh (ead::NodeptrT<double> x)
{
	return age::tanh(x);
}

ead::NodeptrT<double> softmax (ead::NodeptrT<double> x)
{
	auto num = age::exp(x);
	auto denom = age::reduce_sum(num);
	ade::Shape shape = x->shape();
	return age::div(num, age::extend(denom, 0,
		std::vector<uint8_t>(shape.begin(), shape.end())));
}

}

#endif
