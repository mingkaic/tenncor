/*!
 *
 *  tensor.hpp
 *  cnnet
 *
 *  Purpose:
 *  tensor object manages shape and raw data
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#pragma once
#ifndef TENNCOR_TENSOR_HPP
#define TENNCOR_TENSOR_HPP

#include <stdexcept>
#include <string>
#include <type_traits>
#include <cstring>

#include "include/memory/default_alloc.hpp"
#include "include/tensor/tensorshape.hpp"

namespace nnet
{

//! extremely common function fitting source data represented by inshape to
//! destination data represented by outshape
//! data not covered in source are padded with zero
template <typename T>
void fit_toshape (T* dest, const tensorshape& outshape, const T* src, const tensorshape& inshape);

template <typename T>
class tensor // todo: make final, and make mock_tensor compose tensor instance
{
public:
	//! create a rank 0 tensor and specific allocator
	tensor (void);

	//! Create a scalar tensor and specific allocator
	tensor (T scalar, size_t alloc_id = default_alloc::alloc_id);

	//! create a tensor of a specified shape and allocator
	//! if the shape is fully defined, then raw data is allocated
	//! otherwise, tensor will wait for a defined shape
	tensor (tensorshape shape, size_t alloc_id = default_alloc::alloc_id);

	//! deallocate tensor
	virtual ~tensor (void);

	// >>>> COPY && MOVE <<<<
	//! copy constructor
	tensor (const tensor<T>& other, bool shapeonly = false);

	//! move constructor
	tensor (tensor<T>&& other);

	//! copy assignment
	tensor<T>& operator = (const tensor<T>& other);

	//! move assignment
	tensor<T>& operator = (tensor<T>&& other);

	// >>>> ACCESSORS <<<<
	// >>> SHAPE INFORMATION <<<
	//! get tensor shape (allocated if so, allowed shape otherwise)
	tensorshape get_shape (void) const;

	//! get the amount of T elements allocated
	//! if uninitialized, return 0
	size_t n_elems (void) const;

	// >>> DATA INFORMATION <<<
	//! checks if memory is allocated
	bool is_alloc (void) const;

	//! get bytes allocated
	size_t total_bytes (void) const;

	//! get data at coordinate specified
	//! getting out of bound will throw out_of_range error
	//! coordinate values not specified are implied as 0
	T get (std::vector<size_t> coord) const;

	//! exposing unallocated shape will cause assertion death
	//! otherwise return data array copy
	std::vector<T> expose (void) const;

	// >>>> MUTATOR <<<<
	//! get allocator from factory and set it as alloc_
	void set_allocator (size_t alloc_id);

	//! set a new allowed shape
	//! chop raw data outside of new shape
	//! worst case runtime: O(min(N, M))
	//! where N is the original shape size
	//! and M is the resulting shape size
	//! result is shape is compatible with allowed shape
	void set_shape (tensorshape shape);

	//! allocate raw data using allowed (innate) shape
	//! return true if successful
	bool allocate (void);

	//! forcefully deallocate raw_data,
	//! invalidates allocated (external) shape
	//! could be useful when we want to preserve allowed shape
	//! since get_shape when allocated gives allocated shape
	bool deallocate (void);

	//! allocate raw data using input shape
	//! if shape is compatible with allowed
	//! else return false
	bool allocate (const tensorshape shape);

	//! copy raw_data from other expanded/compressed to input shape
	//! allowed shape will be adjusted similar to set_shape
	bool copy_from (const tensor& other, const tensorshape shape);

	// slice along the first dimension
	void slice (size_t dim_start, size_t limit);

	// bool shares_buffer_with (const tensor& other) const;
	// size_t buffer_hash (void) const;

protected:
	// >>>> PROTECTED MEMBERS <<<<
	//! raw data is available to tensor manipulators
	T* raw_data_ = nullptr;

	//! not necessarily defined shape
	tensorshape allowed_shape_;

	//! allocated shape (must be defined)
	tensorshape alloced_shape_;

	friend class itensor;

private:
	//! copy utility helper
	void copy_helper (const tensor<T>& other, bool shapeonly);

	//! move utility helper
	void move_helper (tensor<T>&& other);

	// >>>> PRIVATE MEMBERS <<<<
	//! allocator
	iallocator* alloc_;
};

}

#include "src/tensor/tensor.ipp"

#endif /* TENNCOR_TENSOR_HPP */
