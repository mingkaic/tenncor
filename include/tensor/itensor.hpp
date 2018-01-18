/*!
 *
 *  itensor.hpp
 *  cnnet
 *
 *  Purpose:
 *  encapsulate tensor data type information
 *  and provide type-generic interface
 *
 *  Created by Mingkai Chen on 2017-03-10.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#pragma once
#ifndef TENNCOR_ITENSOR_HPP
#define TENNCOR_ITENSOR_HPP

#include <vector>
#include <cstddef>

#include "proto/serial/tenncor.pb.h"
#include "include/tensor/tensor.hpp"

namespace nnet
{

class itensor
{
public:
	virtual ~itensor (void) {}

	//! clone function
	itensor* clone (bool shapeonly = false) const;

	//! clone function
	itensor* move (void);

	virtual tenncor::tensor_proto::tensor_t get_type (void) const = 0;

	// >>> SHAPE INFORMATION <<<
	//! get tensor shape (allocated if so, allowed shape otherwise)
	virtual tensorshape get_shape (void) const = 0;

	// >> SHAPE UTILITY <<
	//! get the amount of T elements allocated
	//! if uninitialized, return 0
	virtual size_t n_elems (void) const = 0;

	//! get the tensor rank, number of dimensions
	size_t rank (void) const;

	//! get vector dimension values
	std::vector<size_t> dims (void) const;

	// >> SHAPE COMPATIBILITY <<
	//! checks if input tensor has a compatible allowed tensorshape
	//! or if both this and other are allocated and the trimmed shapes are compatible
	bool is_same_size (const itensor& other) const;

	//! check if other tensor's data is compatible with this shape
	bool is_compatible_with (const itensor& other) const;

	//! check if input is compatible with tensor shape
	//! data is compatible if data.size() == (innate or external) shape size
	bool is_compatible_with (size_t ndata) const;

	//! check if an array that is the size of vector 
	//! specified in input is compatible with tensorshape
	//! data is loosely compatible if ndata < (innate or external) shape size
	bool is_loosely_compatible_with (size_t ndata) const;

	//! return compatible shape with n_elems == data.size()
	//! or undefined if compatibility is impossible
	// implementation detail:
	// this algorithm attempts to cover up the first unknown with data.size() / n_known
	// iff data.size() % n_known == 0
	// todo: attempt to add lambda function as parameter to distribute data.size() / n_known among unknowns (same for loosely guess)
	optional<tensorshape> guess_shape (size_t ndata) const;

	//! return loosely compatible shape with n_elems <= data.size()
	//! or undefined if compatibility is impossible
	optional<tensorshape> loosely_guess_shape (size_t ndata) const;

	//! checks if tensorshape is aligned
	//! same number of column for each row
	virtual bool is_aligned (void) const;

	// >>>> DATA INFORMATION <<<<
	//! checks if memory is allocated
	virtual bool is_alloc (void) const = 0;

	//! get bytes allocated
	virtual size_t total_bytes (void) const = 0;

	// >>>> SERIALIZATION HELPERS <<<<
	//! serialize protobuf tensor
	void serialize (tenncor::tensor_proto* proto) const;

	//! read data and shape from other, take allocator as is
	bool from_proto (const tenncor::tensor_proto& other);

	//! read data and shape from other, reassign allocator
	bool from_proto (const tenncor::tensor_proto& other, size_t alloc_id);

	// >>>> DATA MUTATOR <<<<
	//! get allocator from factory and set it as alloc_
	virtual void set_allocator (size_t alloc_id) = 0;

	//! set a new allowed shape
	//! chop raw data outside of new shape
	//! worst case runtime: O(min(N, M))
	//! where N is the original shape size
	//! and M is the resulting shape size
	//! result is shape is compatible with allowed shape
	virtual void set_shape (tensorshape shape) = 0;

	//! allocate raw data using innate shape
	virtual bool allocate (void) = 0;

	//! forcefully deallocate raw_data, invalidates external shape
	virtual bool deallocate (void) = 0;

	//! allocate raw data using input shape
	//! if shape is compatible with allowed
	//! else return false
	virtual bool allocate (const tensorshape shape) = 0;

	//! copy raw_data from other expanded/compressed to input shape
	//! allowed shape will be adjusted similar to set_shape
	virtual bool copy_from (const itensor& other, const tensorshape shape) = 0;

	// slice along the first dimension
	virtual void slice (size_t dim_start, size_t limit) = 0;

protected:
	// >>>> ABSTRACT CLONE <<<<
	//! clone implementation
	virtual itensor* clone_impl (bool shapeonly) const = 0;
	
	//! move implementation
	virtual itensor* move_impl (void) = 0;

	template <typename T>
	void set_data (tensor<T>& tens, void* data, size_t nbytes)
	{
		std::memcpy(tens.raw_data_, data, nbytes);
	}

	template <typename T>
	void set_allowed_shape (tensor<T>& tens, const tensorshape& shape)
	{
		tens.allowed_shape_ = shape;
	}

	template <typename T>
	void set_alloced_shape (tensor<T>& tens, const tensorshape& shape)
	{
		tens.alloced_shape_ = shape;
	}

	template <typename T>
	void* get_data (tensor<T>& tens) const
	{
		return (void*) tens.raw_data_;
	}

	template <typename T>
	const void* get_data (const tensor<T>& tens) const
	{
		return (const void*) tens.raw_data_;
	}

	template <typename T>
	const tensorshape& get_allowed_shape (const tensor<T>& tens) const
	{
		return tens.allowed_shape_;
	}

	template <typename T>
	const tensorshape& get_alloced_shape (const tensor<T>& tens) const
	{
		return tens.alloced_shape_;
	}

	virtual void set_data (void* data, size_t nbytes) = 0;

	virtual void set_allowed_shape (const tensorshape& shape) = 0;

	virtual void set_alloced_shape (const tensorshape& shape) = 0;

	virtual void* get_data (void) = 0;

	virtual const void* get_data (void) const = 0;

	virtual const tensorshape& get_allowed_shape (void) const = 0;

	virtual const tensorshape& get_alloced_shape (void) const = 0;

	friend class itensor_handler;
};

size_t type_size (tenncor::tensor_proto::tensor_t type);

template <typename T>
tenncor::tensor_proto::tensor_t get_prototype (void)
{
	throw std::exception(); // unsupported type
}

template <>
tenncor::tensor_proto::tensor_t get_prototype<double> (void);

template <>
tenncor::tensor_proto::tensor_t get_prototype<signed> (void);

}

#endif /* TENNCOR_ITENSOR_HPP */
