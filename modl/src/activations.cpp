#include "modl/activations.hpp"

#ifdef MODL_ACTIVATIONS_HPP

namespace modl
{

LayerptrT ActivationBuilder::build (void) const
{
	return std::make_shared<Activation>(label_, act_type_);
}

LayerptrT sigmoid (std::string label)
{
	return std::make_shared<Activation>(label, sigmoid_layer_key);
}

LayerptrT tanh (std::string label)
{
	return std::make_shared<Activation>(label, tanh_layer_key);
}

}

#endif
