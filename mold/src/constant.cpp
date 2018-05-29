//
//  constant.cpp
//  mold
//

#include "mold/constant.hpp"
#include "mold/functor.hpp"

#ifdef MOLD_CONSTANT_HPP

namespace mold
{

Constant::Constant (std::shared_ptr<char> data,
	clay::Shape shape, clay::DTYPE type) :
state_(data, shape, type), data_(data)
{
	if (nullptr == data ||
		false == shape.is_fully_defined() ||
		clay::DTYPE::BAD == type)
	{
		throw std::exception(); // todo: add context
	}
}

bool Constant::has_data (void) const
{
	return true;
}

clay::State Constant::get_state (void) const
{
	return state_;
}

}

#endif
