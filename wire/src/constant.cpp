//
//  constant.cpp
//  wire
//

#include <cassert>

#include "wire/constant.hpp"

#ifdef WIRE_CONSTANT_HPP

namespace wire
{

Constant::Constant (std::shared_ptr<char> data, clay::Shape shape,
	clay::DTYPE dtype, std::string label, Graph& graph) :
	Identifier(&graph, new mold::Constant(data, shape, dtype), label) {}

}

#endif
