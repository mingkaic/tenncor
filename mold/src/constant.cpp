//
//  constant.cpp
//  mold
//

#include "mold/constant.hpp"
#include "mold/functor.hpp"
#include "mold/error.hpp"

#include "clay/memory.hpp"
#include "clay/error.hpp"

#ifdef MOLD_CONSTANT_HPP

namespace mold
{

Constant::Constant (std::shared_ptr<char> data,
	clay::Shape shape, clay::DTYPE type) :
state_(data, shape, type), data_(data)
{
	if (nullptr == data)
	{
		throw NilDataError();
	}
	if (false == shape.is_fully_defined())
	{
		throw clay::InvalidShapeError(shape);
	}
	if (clay::BAD == type)
	{
		throw clay::UnsupportedTypeError(type);
	}
}

Constant::Constant (const Constant& other)
{
	size_t nbytes = other.state_.shape_.n_elems() * clay::type_size(other.state_.dtype_);
	data_ = clay::make_char(nbytes);
	std::memcpy(data_.get(), other.data_.get(), nbytes);
	state_ = clay::State(data_, other.state_.shape_, other.state_.dtype_);
}

bool Constant::has_data (void) const
{
	return true;
}

clay::Shape Constant::get_shape (void) const
{
	return state_.shape_;
}

clay::State Constant::get_state (void) const
{
	return state_;
}

iNode* Constant::clone_impl (void) const
{
	return new Constant(*this);
}

}

#endif
