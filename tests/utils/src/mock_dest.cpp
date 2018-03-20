#include "mock_dest.hpp"

#ifdef TTEST_MOCK_DEST_HPP

namespace testutils
{

mock_data_dest::mock_data_dest (void) : result_(16, ' ') {}

void mock_data_dest::set_data (std::weak_ptr<void> data, TENS_TYPE type, nnet::tensorshape shape, size_t idx)
{
	label_incr("set_data");
	std::stringstream ss;
	ss << idx;
	set_label("set_data", ss.str());

	std::memcpy(&result_[0], data.lock().get(), 16);
	type_ = type;
	shape_ = shape;
}

}

#endif
