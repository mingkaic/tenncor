//
//  state.cpp
//  clay
//

#include "clay/state.hpp"

#ifdef CLAY_STATE_HPP

namespace clay
{

State::State (void) = default;

State::State (std::weak_ptr<char> data, Shape shape, DTYPE dtype) :
	shape_(shape), dtype_(dtype), data_(data) {}

char* State::get (void) const
{
	char* out = nullptr;
	if (false == data_.expired())
	{
		out = data_.lock().get();
	}
	return out;
}

}

#endif
