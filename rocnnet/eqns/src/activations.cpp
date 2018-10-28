#include "rocnnet/eqns/activations.hpp"

llo::DataNode sigmoid (llo::DataNode x)
{
	ade::Shape shape = x.tensor_->shape();
	auto denom = llo::add(llo::one(shape), llo::exp(llo::neg(x)));
	return llo::div(llo::one(shape), denom);
}

llo::DataNode tanh (llo::DataNode x)
{
	ade::Shape shape = x.tensor_->shape();
	auto expxx = llo::exp(llo::add(x, x));
	auto num = llo::add(expxx, llo::one(shape));
	auto denom = llo::sub(expxx, llo::one(shape));
	return llo::div(num, denom);
}

llo::DataNode softmax (llo::DataNode x)
{
	auto num = llo::exp(x);
	auto denom = llo::reduce_sum(num);
	return llo::div(num, denom);
}
