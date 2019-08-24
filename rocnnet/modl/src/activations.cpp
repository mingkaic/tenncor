#include "rocnnet/modl/activations.hpp"

#ifdef MODL_ACTIVATIONS_HPP

namespace modl
{

LayerptrT ActivationBuilder::build (void) const
{
	return std::make_shared<ActivationLayer>(label_, act_type_);
}

LayerptrT sigmoid (std::string label)
{
	return std::make_shared<ActivationLayer>(label, sigmoid_layer_key);
}

LayerptrT tanh (std::string label)
{
	return std::make_shared<ActivationLayer>(label, tanh_layer_key);
}

}

#endif
