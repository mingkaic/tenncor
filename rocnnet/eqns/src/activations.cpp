#include "rocnnet/eqns/activations.hpp"

llo::DataNode sigmoid (llo::DataNode x)
{
	auto denom = llo::add(llo::one(), llo::exp(llo::neg(x)));
	return llo::div(llo::one(), denom);
}

llo::DataNode tanh (llo::DataNode x)
{
	auto expxx = llo::exp(llo::add(x, x));
	auto num = llo::add(expxx, llo::one());
	auto denom = llo::sub(expxx, llo::one());
	return llo::div(num, denom);
}

llo::DataNode softmax (llo::DataNode x)
{
	auto num = llo::exp(x);
	auto denom = llo::reduce_sum(num);
	return llo::div(num, denom);
}
