#include "modl/rbm.hpp"

#ifdef MODL_RBM_HPP

namespace modl
{

LayerptrT RBMBuilder::build (void) const
{
	if (3 != layers_.size())
	{
		logs::fatalf("cannot make rbm without hidden, visible, "
			"and activation layer, got %d layers",
			layers_.size());
	}
	return std::make_shared<RBM>(
		std::static_pointer_cast<Dense>(layers_[0]),
		std::static_pointer_cast<Dense>(layers_[1]),
		std::static_pointer_cast<Activation>(layers_[2]),
		label_);
}

}

#endif
