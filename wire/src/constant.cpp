//
//  constant.cpp
//  wire
//

#include <cassert>

#include "wire/constant.hpp"
#include "wire/operators.hpp"

#ifdef WIRE_CONSTANT_HPP

namespace wire
{

Constant::Constant (std::shared_ptr<char> data, clay::Shape shape,
	clay::DTYPE dtype, std::string label, Graph& graph) :
	Identifier(&graph, new mold::Constant(data, shape, dtype), label) {}

Identifier* Constant::derive (Identifier* wrt)
{
	if (wrt == this)
	{
		throw std::logic_error("deriving with respect to a constant");
	}
	// clay::State state = get()->get_state();
	// return make_zero(state.shape_, state.dtype_);
	return sub(this, this);
}

}

#endif
