#include "rocnnet/eqns/helper.hpp"

#ifdef EQNS_HELPER_HPP

namespace eqns
{

ead::NodeptrT<double> one_binom (ead::NodeptrT<double> a)
{
	const ade::Shape& shape = a->get_tensor()->shape();
	auto trial = age::rand_unif(
		ead::convert_to_node(ead::make_variable_scalar<double>(0.0, shape)),
		ead::convert_to_node(ead::make_variable_scalar<double>(1.0, shape)));
	return age::lt(trial, a);
}

ead::NodeptrT<double> weighed_bias_add (ead::NodeptrT<double> weighed,
    ead::NodeptrT<double> bias)
{
	const ade::Shape& shape = weighed->get_tensor()->shape();
	return age::add(weighed, age::extend(bias, 1, {shape.at(1)}));
}

}

#endif
