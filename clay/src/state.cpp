//
//  state.cpp
//  clay
//

#include "clay/state.hpp"

#ifdef CLAY_STATE_HPP

namespace clay
{

State::State (std::weak_ptr<const char> data, Shape shape, DTYPE dtype) : 
	data_(data), shape_(shape), dtype_(dtype) {}

}

#endif
