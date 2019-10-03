#include "layr/seqmodel.hpp"

#ifdef LAYR_SEQMODEL_HPP

namespace layr
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
