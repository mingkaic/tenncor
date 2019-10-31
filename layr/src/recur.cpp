#include "layr/recur.hpp"

#ifdef LAYR_RECUR_HPP

namespace layr
{

LayerptrT RecurBuilder::build (void) const
{
	if (2 != layers_.size())
	{
		logs::fatalf("cannot make recurrent without dense cell, "
			"and activation layer, got %d layers",
			layers_.size());
	}
	return std::make_shared<Recur>(
		std::static_pointer_cast<Dense>(layers_[0]),
		std::static_pointer_cast<ULayer>(layers_[1]),
        init_state_,
		label_);
}

}

#endif
