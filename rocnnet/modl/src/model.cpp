#include "rocnnet/modl/model.hpp"

#ifdef MODL_MODEL_HPP

namespace modl
{

LayerptrT SeqModelBuilder::build (void) const
{
	auto model = std::make_shared<SequentialModel>(label_);
	for (auto& layer : layers_)
	{
		model->push_back(layer);
	}
	return model;
}

}

#endif
