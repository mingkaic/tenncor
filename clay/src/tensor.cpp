//
//  clay.ipp
//  cnnet
//

#include "clay/tensor.hpp"

#ifdef TENSOR_TENSOR_HPP

namespace clay
{

Tensor::Tensor (std::shared_ptr<char> data, Shape shape, DTYPE dtype) :
	data_(data), shape_(shape), dtype_(dtype)
{
	if (nullptr == data)
	{
		throw std::exception(); // todo: add context
	}
	if (false == shape.is_fully_defined())
	{
		throw std::exception(); // todo: add context
	}
	if (DTYPE::BAD == dtype)
	{
		throw std::exception(); // todo: add context
	}
}

Tensor::Tensor (Tensor&& other) :
	data_(std::move(other.data_)),
	shape_(std::move(other.shape_)),
	dtype_(other.dtype_)
{
	other.dtype_ = DTYPE::BAD;
}

Tensor& Tensor::operator = (Tensor&& other)
{
	if (this != &other)
	{
		data_ = std::move(other.data_);
		shape_ = std::move(other.shape_);
		dtype_ = other.dtype_;
		other.dtype_ = DTYPE::BAD;
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

bool Tensor::read_from (const iSource& src)
{
	bool successful = data_ != nullptr;
	if (successful)
	{
		State state = src.get_data();
		successful = false == state.data_.expired() &&
			shape_.is_compatible_with(state.shape_) &&
			state.shape_.is_fully_defined() &&
			state.dtype_ == dtype_;
		if (successful)
		{
			std::memcpy(data_.get(), state.data_.lock().get(), total_bytes());
		}
	}
	return successful;
}

}

#endif
