#include "layr/activations.hpp"

#ifdef LAYR_ACTIVATIONS_HPP

namespace layr
{

LayerptrT ActivationBuilder::build (void) const
{
	return std::make_shared<Activation>(act_type_, label_);
}

NodeptrT softmax_from_layer (const Activation& layer,
	NodeptrT input)
{
	return tenncor::softmax<PybindT>(input,
		std::stoi(layer.get_label()), 1);
}

ActivationptrT sigmoid (void)
{
	return std::make_shared<Activation>(sigmoid_layer_key);
}

ActivationptrT tanh (void)
{
	return std::make_shared<Activation>(tanh_layer_key);
}

ActivationptrT softmax (teq::RankT dim)
{
	return std::make_shared<Activation>(softmax_layer_key,
		fmts::sprintf("%d", dim));
}

}

#endif
