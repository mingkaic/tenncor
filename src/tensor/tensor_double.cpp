//
//  tensor_double.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-15.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/tensor/tensor_double.hpp"

#ifdef TENNCOR_TENSOR_DOUBLE_HPP

namespace nnet
{

tensor_double::tensor_double (void) {}

tensor_double::tensor_double (double scalar, size_t alloc_id) : 
	content_(scalar, alloc_id) {}

tensor_double::tensor_double (tensorshape shape, size_t alloc_id) : 
	content_(shape, alloc_id) {}

tensor_double* tensor_double::clone (bool shapeonly) const
{
	return static_cast<tensor_double*>(clone_impl(shapeonly));
}

tensor_double* tensor_double::move (void)
{
	return static_cast<tensor_double*>(move_impl());
}

tenncor::tensor_proto::tensor_t tensor_double::get_type (void) const
{
	return tenncor::tensor_proto::DOUBLE_T;
}

tensorshape tensor_double::get_shape (void) const
{
	return content_.get_shape();
}

size_t tensor_double::n_elems (void) const
{
	return content_.n_elems();
}

bool tensor_double::is_alloc (void) const
{
	return content_.is_alloc();
}

size_t tensor_double::total_bytes (void) const
{
	return content_.total_bytes();
}

std::vector<double> tensor_double::expose (void) const
{
	return content_.expose();
}

double tensor_double::get (std::vector<size_t> coord) const
{
	return content_.get(coord);
}

void tensor_double::set_allocator (size_t alloc_id)
{
	return content_.set_allocator(alloc_id);
}

void tensor_double::set_shape (tensorshape shape)
{
	return content_.set_shape(shape);
}

bool tensor_double::allocate (void)
{
	return content_.allocate();
}

bool tensor_double::deallocate (void)
{
	return content_.deallocate();
}

bool tensor_double::allocate (const tensorshape shape)
{
	return content_.allocate(shape);
}

bool tensor_double::copy_from (const itensor& other, const tensorshape shape)
{
	if (const tensor_double* tens = dynamic_cast<const tensor_double*>(&other))
	{
		return content_.copy_from(tens->content_, shape);
	}
	return false;
}

void tensor_double::slice (size_t dim_start, size_t limit)
{
	content_.slice(dim_start, limit);
}

tensor_double::tensor_double (const tensor_double& other, bool shapeonly) :
	content_(other.content_, shapeonly) {}

itensor* tensor_double::clone_impl (bool shapeonly) const
{
	return new tensor_double(*this, shapeonly);
}

itensor* tensor_double::move_impl (void)
{
	return new tensor_double(std::move(*this));
}

void tensor_double::set_data (void* data, size_t nbytes)
{
	itensor::template set_data<double>(content_, data, nbytes);
}

void tensor_double::set_allowed_shape (const tensorshape& shape)
{
	itensor::template set_allowed_shape<double>(content_, shape);
}

void tensor_double::set_alloced_shape (const tensorshape& shape)
{
	itensor::template set_alloced_shape<double>(content_, shape);
}

void* tensor_double::get_data (void)
{
	return itensor::template get_data<double>(content_);
}

const void* tensor_double::get_data (void) const
{
	return itensor::template get_data<double>(content_);
}

const tensorshape& tensor_double::get_allowed_shape (void) const
{
	return itensor::template get_allowed_shape<double>(content_);
}

const tensorshape& tensor_double::get_alloced_shape (void) const
{
	return itensor::template get_alloced_shape<double>(content_);
}

}

#endif
