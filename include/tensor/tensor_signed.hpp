/*!
 *
 *  tensor_signed.hpp
 *  cnnet
 *
 *  Purpose:
 *  signed int implementation of tensor
 *
 *  Created by Mingkai Chen on 2017-03-10.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#pragma once
#ifndef TENNCOR_TENSOR_SIGNED_HPP
#define TENNCOR_TENSOR_SIGNED_HPP

#include <vector>
#include <cstddef>

#include "include/tensor/itensor.hpp"

namespace nnet
{

class tensor_signed : public itensor
{
public:
	//! create a rank 0 tensor and specific allocator
	tensor_signed (void);

	//! Create a scalar tensor and specific allocator
	tensor_signed (signed scalar, size_t alloc_id = default_alloc::alloc_id);

	//! create a tensor of a specified shape and allocator
	//! if the shape is fully defined, then raw data is allocated
	//! otherwise, tensor will wait for a defined shape
	tensor_signed (tensorshape shape, size_t alloc_id = default_alloc::alloc_id);

	//! clone function
	tensor_signed* clone (bool shapeonly = false) const;

	//! clone function
	tensor_signed* move (void);

	//! copy assignment
	tensor_signed& operator = (const tensor_signed& other);

	//! move assignment
	tensor_signed& operator = (tensor_signed&& other);

	virtual tenncor::tensor_proto::tensor_t get_type (void) const;

	// >>> SHAPE INFORMATION <<<
	//! get tensor shape (allocated if so, allowed shape otherwise)
	virtual tensorshape get_shape (void) const;

	//! get the amount of T elements allocated
	//! if uninitialized, return 0
	virtual size_t n_elems (void) const;

	// >>>> DATA INFORMATION <<<<
	//! checks if memory is allocated
	virtual bool is_alloc (void) const;

	//! get bytes allocated
	virtual size_t total_bytes (void) const;

	//! exposing unallocated shape will cause assertion death
	//! otherwise return data array copy
	std::vector<signed> expose (void) const;

	//! get data at coordinate specified
	signed get (std::vector<size_t> coord) const;

	// >>>> DATA MUTATOR <<<<
	//! get allocator from factory and set it as alloc_
	virtual void set_allocator (size_t alloc_id);

	//! set a new allowed shape
	//! chop raw data outside of new shape
	//! worst case runtime: O(min(N, M))
	//! where N is the original shape size
	//! and M is the resulting shape size
	//! result is shape is compatible with allowed shape
	virtual void set_shape (tensorshape shape);

	//! allocate raw data using innate shape
	virtual bool allocate (void);

	//! forcefully deallocate raw_data, invalidates external shape
	virtual bool deallocate (void);

	//! allocate raw data using input shape
	//! if shape is compatible with allowed
	//! else return false
	virtual bool allocate (const tensorshape shape);

	//! copy raw_data from other expanded/compressed to input shape
	//! allowed shape will be adjusted similar to set_shape
	virtual bool copy_from (const itensor& other, const tensorshape shape);

	// slice along the first dimension
	virtual void slice (size_t dim_start, size_t limit);

protected:
	// >>>> COPY && MOVE <<<<
	//! copy constructor
	tensor_signed (const tensor_signed& other, bool shapeonly = false);

	//! move constructor
	tensor_signed (tensor_signed&& other);

	// >>>> ABSTRACT CLONE <<<<
	//! clone implementation
	virtual itensor* clone_impl (bool shapeonly) const;
	
	//! move implementation
	virtual itensor* move_impl (void);

	virtual void set_data (void* data, size_t nbytes);

	virtual void set_allowed_shape (const tensorshape& shape);

	virtual void set_alloced_shape (const tensorshape& shape);

	virtual void* get_data (void);

	virtual const void* get_data (void) const;

	virtual const tensorshape& get_allowed_shape (void) const;

	virtual const tensorshape& get_alloced_shape (void) const;

private:
	tensor<signed> content_;
};

}

#endif /* TENNCOR_TENSOR_SIGNED_HPP */
