//
//  state.cpp
//  clay
//

#include "clay/state.hpp"

#ifdef CLAY_STATE_HPP

namespace clay
{

State::State (std::weak_ptr<char> data, Shape shape, DTYPE dtype) :
	shape_(shape), dtype_(dtype), block_(data)
{
	if (false == block_.expired())
	{
		data_ = block_.lock().get();
	}
}

State::State (char* data, std::weak_ptr<char> block,
	Shape shape, DTYPE dtype) :
	shape_(shape), dtype_(dtype), data_(data), block_(block) {}

State::State (const State& other, std::weak_ptr<char> block) :
	shape_(other.shape_), dtype_(other.dtype_), block_(block)
{
	if (false == block_.expired())
	{
		data_ = block.lock().get();
	}
}

char* State::get (void) const
{
	char* out = nullptr;
	if (false == block_.expired())
	{
		out = data_;
	}
	return out;
}

}

#endif
