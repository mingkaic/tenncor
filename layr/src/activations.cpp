#include "layr/activations.hpp"

#ifdef LAYR_ACTIVATIONS_HPP

namespace layr
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
