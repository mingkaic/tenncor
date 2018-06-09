//
//  clay.ipp
//  cnnet
//

#include "clay/tensor.hpp"
#include "clay/memory.hpp"
#include "clay/error.hpp"

#ifdef CLAY_TENSOR_HPP

namespace clay
{

Tensor::Tensor (Shape shape, DTYPE dtype) :
	shape_(shape), dtype_(dtype)
{
	if (false == shape.is_fully_defined())
	{
		throw InvalidShapeError(shape);
	}
	if (DTYPE::BAD == dtype)
	{
		throw UnsupportedTypeError(dtype);
	}
	size_t nbytes = shape.n_elems() * type_size(dtype);
	data_ = make_char(nbytes);
}

Tensor::Tensor (const Tensor& other)
{
	size_t obytes = other.total_bytes();
	data_ = make_char(obytes);
	std::memcpy(data_.get(), other.data_.get(), obytes);

	shape_ = other.shape_;
	dtype_ = other.dtype_;
}

Tensor& Tensor::operator = (const Tensor& other)
{
	if (this != &other)
	{
		size_t obytes = other.total_bytes();
		size_t nbytes = total_bytes();
		if (nbytes != obytes)
		{
			data_ = make_char(obytes);
		}
		std::memcpy(data_.get(), other.data_.get(), obytes);

		shape_ = other.shape_;
		dtype_ = other.dtype_;
	}
	return *this;
}

State Tensor::get_state (void) const
{
	return State{data_, shape_, dtype_};
}

Shape Tensor::get_shape (void) const
{
	return shape_;
}

DTYPE Tensor::get_type (void) const
{
	return dtype_;
}

size_t Tensor::total_bytes (void) const
{
	return shape_.n_elems() * type_size(dtype_);
}

}

#endif
