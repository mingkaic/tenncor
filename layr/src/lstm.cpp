#include "layr/lstm.hpp"

#ifdef LAYR_LSTM_HPP

namespace layr
{

LayerptrT LSTMBuilder::build (void) const
{
	if (4 != layers_.size())
	{
		logs::fatalf("cannot make lstm without gate, forget, "
			"ingate, and outgate layer, got %d layers",
			layers_.size());
	}
	return std::make_shared<LSTM>(
		std::static_pointer_cast<Dense>(layers_[0]),
		std::static_pointer_cast<Dense>(layers_[1]),
		std::static_pointer_cast<Dense>(layers_[2]),
		std::static_pointer_cast<Dense>(layers_[3]),
		label_);
}

}

#endif
