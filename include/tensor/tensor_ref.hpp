/*  tensor_ref.hpp
 *  cnnet
 *
 *  Purpose:
 *  tensor containing argument reference pointers 
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/tensor.hpp"

#pragma once
#ifndef TENNCOR_TENSOR_REF_HPP
#define TENNCOR_TENSOR_REF_HPP

namespace nnet
{

struct addr
{
	inode* node_;
	size_t idx_;

	void* get_ptr (void)
	{

	}
};

struct ref
{
	std::vector<addr> srcs_;
	TENS_TYPE dtype_;
};

class tensor_ref : public tensor
{
public:
	//! create a rank 0 tensor and specific allocator
	tensor_ref (void) {}

	//! Create a scalar tensor and specific allocator
	tensor_ref (ref scalar, size_t alloc_id = default_alloc::alloc_id) :
		content_(scalar, alloc_id)
	{
		working_mem_ = std::malloc(sizeof(double));
	}

	//! create a tensor of a specified shape and allocator
	//! if the shape is fully defined, then raw data is allocated
	//! otherwise, tensor will wait for a defined shape
	tensor_ref (tensorshape shape, size_t alloc_id = default_alloc::alloc_id) :
		content_(shape, alloc_id)
	{
		working_mem_ = std::malloc(sizeof(double) * shape.n_elems());
	}

	~tensor_ref (void)
	{
		std::free(working_mem_);
	}

	//! clone function
	tensor_ref clone (bool shapeonly = false) const
	{
		return static_cast<tensor_ref>(this->clone_impl(shapeonly));
	}

	//! clone function
	tensor_ref move (void)
	{
		return static_cast<tensor_ref>(this->move_impl());
	}

	//! copy assignment
	tensor_ref& operator = (const tensor_ref& other)
	{
		if (&other != this)
		{
			content_ = other.content_;
		}
		return *this;
	}

	//! move assignment
	tensor_ref& operator = (tensor_ref&& other)
	{
		if (&other != this)
		{
			content_ = std::move(other.content_);
		}
		return *this;
	}

	ref* get_refs (void) const
	{
		return (ref*) get_data();
	}

	virtual TENS_TYPE get_type (void) const
	{
		return REF;
	}

	// >>> SHAPE INFORMATION <<<
	//! get tensor shape (allocated if so, allowed shape otherwise)
	virtual tensorshape get_shape (void) const
	{
		return content_.get_shape();
	}

	//! get the amount of T elements allocated
	//! if uninitialized, return 0
	virtual size_t n_elems (void) const
	{
		return content_.n_elems();
	}

	// >>>> DATA INFORMATION <<<<
	//! checks if memory is allocated
	virtual bool is_alloc (void) const
	{
		return content_.is_alloc();
	}

	//! get bytes allocated
	virtual size_t total_bytes (void) const
	{
		return content_.total_bytes();
	}

	//! exposing unallocated shape will cause assertion death
	//! otherwise return data array copy
	virtual std::vector<double> expose (void) const
	{
		return std::vector<double>(working_mem_, working_mem_ + n_elems());
	}

	//! get data at coordinate specified
	virtual double get (std::vector<size_t> coord) const
	{
		return content_.get(coord);
	}

	// >>>> DATA MUTATOR <<<<
	//! get allocator from factory and set it as alloc_
	virtual void set_allocator (size_t alloc_id)
	{
		return content_.set_allocator(alloc_id);
	}

	//! set a new allowed shape
	//! chop raw data outside of new shape
	//! worst case runtime: O(min(N, M))
	//! where N is the original shape size
	//! and M is the resulting shape size
	//! result is shape is compatible with allowed shape
	virtual void set_shape (tensorshape shape)
	{
		return content_.set_shape(shape);
	}

	//! allocate raw data using innate shape
	virtual bool allocate (void)
	{
		return content_.allocate();
	}

	//! forcefully deallocate raw_data, invalidates external shape
	virtual bool deallocate (void)
	{
		return content_.deallocate();
	}

	//! allocate raw data using input shape
	//! if shape is compatible with allowed
	//! else return false
	virtual bool allocate (const tensorshape shape)
	{
		working_mem_ = malloc(sizeof(double) * shape.n_elems());
		return content_.allocate(shape);
	}

	//! copy raw_data from other expanded/compressed to input shape
	//! allowed shape will be adjusted similar to set_shape
	virtual bool copy_from (const tensor& other, const tensorshape shape)
	{
		if (const tensor_double* tens = dynamic_cast<const tensor_double*>(&other))
		{
			return content_.copy_from(tens->content_, shape);
		}
		return false;
	}

	// slice along the first dimension
	virtual void slice (size_t dim_start, size_t limit)
	{
		content_.slice(dim_start, limit);
	}

	void* working_mem_ = nullptr;

protected:
	// >>>> COPY && MOVE <<<<
	//! copy constructor
	tensor_ref (const tensor_ref& other, bool shapeonly = false);

	//! move constructor
	tensor_ref (tensor_ref&& other);

	// >>>> ABSTRACT CLONE <<<<
	//! clone implementation
	virtual tensor* clone_impl (bool shapeonly) const
	{
		return new tensor_ref(*this, shapeonly);
	}
	
	//! move implementation
	virtual tensor* move_impl (void)
	{
		return new tensor_ref(std::move(*this));
	}

	virtual void set_data (void* data, size_t nbytes)
	{
		std::memcpy(working_mem_, data, nbytes);
	}

	virtual void set_allowed_shape (const tensorshape& shape)
	{
		tensor::template set_allowed_shape<ref>(content_, shape);
	}

	virtual void set_alloced_shape (const tensorshape& shape)
	{
		tensor::template set_alloced_shape<ref>(content_, shape);
	}

	virtual void* get_data (void)
	{
		return working_mem_;
	}

	virtual const void* get_data (void) const
	{
		return working_mem_;
	}

	virtual const tensorshape& get_allowed_shape (void) const
	{
		return tensor::template get_allowed_shape<ref>(content_);
	}

	virtual const tensorshape& get_alloced_shape (void) const
	{
		return tensor::template get_alloced_shape<double>(content_);
	}

private:
	tensor<ref> content_;
};

}

#endif /* TENNCOR_TENSOR_REF_HPP */