//
//  tensor_signed.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-15.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/tensor/tensor_signed.hpp"

#ifdef TENNCOR_TENSOR_SIGNED_HPP

namespace nnet
{

tensor_signed::tensor_signed (void) {}

tensor_signed::tensor_signed (signed scalar, size_t alloc_id) : 
	content_(scalar, alloc_id) {}

tensor_signed::tensor_signed (tensorshape shape, size_t alloc_id) : 
	content_(shape, alloc_id) {}

tensor_signed* tensor_signed::clone (bool shapeonly) const
{
	return static_cast<tensor_signed*>(clone_impl(shapeonly));
}

tensor_signed* tensor_signed::move (void)
{
	return static_cast<tensor_signed*>(move_impl());
}

tenncor::tensor_proto::tensor_t tensor_signed::get_type (void) const
{
	return tenncor::tensor_proto::SIGNED_T;
}

tensorshape tensor_signed::get_shape (void) const
{
	return content_.get_shape();
}

size_t tensor_signed::n_elems (void) const
{
	return content_.n_elems();
}

bool tensor_signed::is_alloc (void) const
{
	return content_.is_alloc();
}

size_t tensor_signed::total_bytes (void) const
{
	return content_.total_bytes();
}

std::vector<signed> tensor_signed::expose (void) const
{
	return content_.expose();
}

signed tensor_signed::get (std::vector<size_t> coord) const
{
	return content_.get(coord);
}

void tensor_signed::set_allocator (size_t alloc_id)
{
	return content_.set_allocator(alloc_id);
}

void tensor_signed::set_shape (tensorshape shape)
{
	return content_.set_shape(shape);
}

bool tensor_signed::allocate (void)
{
	return content_.allocate();
}

bool tensor_signed::deallocate (void)
{
	return content_.deallocate();
}

bool tensor_signed::allocate (const tensorshape shape)
{
	return content_.allocate(shape);
}

bool tensor_signed::copy_from (const itensor& other, const tensorshape shape)
{
	if (const tensor_signed* tens = dynamic_cast<const tensor_signed*>(&other))
	{
		return content_.copy_from(tens->content_, shape);
	}
	return false;
}

void tensor_signed::slice (size_t dim_start, size_t limit)
{
	content_.slice(dim_start, limit);
}

tensor_signed::tensor_signed (const tensor_signed& other, bool shapeonly) :
	content_(other.content_, shapeonly) {}

itensor* tensor_signed::clone_impl (bool shapeonly) const
{
	return new tensor_signed(*this, shapeonly);
}

itensor* tensor_signed::move_impl (void)
{
	return new tensor_signed(std::move(*this));
}

void tensor_signed::set_data (void* data, size_t nbytes)
{
	itensor::template set_data<signed>(content_, data, nbytes);
}

void tensor_signed::set_allowed_shape (const tensorshape& shape)
{
	itensor::template set_allowed_shape<signed>(content_, shape);
}

void tensor_signed::set_alloced_shape (const tensorshape& shape)
{
	itensor::template set_alloced_shape<signed>(content_, shape);
}

void* tensor_signed::get_data (void)
{
	return itensor::template get_data<signed>(content_);
}

const void* tensor_signed::get_data (void) const
{
	return itensor::template get_data<signed>(content_);
}

const tensorshape& tensor_signed::get_allowed_shape (void) const
{
	return itensor::template get_allowed_shape<signed>(content_);
}

const tensorshape& tensor_signed::get_alloced_shape (void) const
{
	return itensor::template get_alloced_shape<signed>(content_);
}

}

#endif
