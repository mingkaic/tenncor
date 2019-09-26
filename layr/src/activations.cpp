#include "layr/activations.hpp"

#ifdef LAYR_ACTIVATIONS_HPP

namespace layr
{

LayerptrT ActivationBuilder::build (void) const
{
	return std::make_shared<Activation>(label_, act_type_);
}

eteq::NodeptrT<PybindT> softmax_from_layer (const Activation& layer,
	eteq::NodeptrT<PybindT> input)
{
	return tenncor::softmax<PybindT>(input,
		std::stoi(layer.get_label()), 1);
}

LayerptrT sigmoid (void)
{
	return std::make_shared<Activation>(sigmoid_layer_key);
}

LayerptrT tanh (void)
{
	return std::make_shared<Activation>(tanh_layer_key);
}

LayerptrT softmax (teq::RankT dim)
{
	return std::make_shared<Activation>(softmax_layer_key,
		fmts::sprintf("%d", dim));
}

}

#endif
