#include "layr/gru.hpp"

#ifdef LAYR_GRU_HPP

namespace layr
{

LayerptrT GRUBuilder::build (void) const
{
	if (3 != layers_.size())
	{
		logs::fatalf("cannot make gru without update, reset, and hidden "
			"gate layers, got %d layers", layers_.size());
	}
	return std::make_shared<GRU>(
		std::static_pointer_cast<Dense>(layers_[0]),
		std::static_pointer_cast<Dense>(layers_[1]),
		std::static_pointer_cast<Dense>(layers_[2]),
		label_);
}

}

#endif
