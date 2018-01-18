//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <cstring>
#include <functional> // for std::bad_function_call();

#include "include/memory/alloc_builder.hpp"

#ifdef TENNCOR_TENSOR_HPP

namespace nnet
{

template <typename T>
void fit_toshape (T* dest, const tensorshape& outshape, const T* src, const tensorshape& inshape)
{
	assert(outshape.is_fully_defined() && inshape.is_fully_defined());
	std::vector<size_t> outlist = outshape.as_list();
	std::vector<size_t> inlist = inshape.as_list();
	size_t numout = outshape.n_elems();

	size_t minrank = std::min(outlist.size(), inlist.size());
	tensorshape clippedshape = inshape.with_rank(minrank);
	size_t numin = clippedshape.n_elems();

	memset(dest, 0, sizeof(T) * numout);
	size_t basewidth = std::min(outlist[0], inlist[0]);
	size_t srcidx = 0;
	while (srcidx < numin)
	{
		// check source index to ensure it is within inlist bounds
		std::vector<size_t> srccoord = clippedshape.coordinate_from_idx(srcidx);
		bool srcinbound = true;
		size_t src_jump = 1;
		for (size_t i = 1, m = minrank; srcinbound && i < m; i++)
		{
			srcinbound = srccoord[i] < outlist[i];
			if (false == srcinbound)
			{
				src_jump *= (inlist[i] - srccoord[i]);
			}
			else
			{
				src_jump *= inlist[i];
			}
		}
		if (false == srcinbound)
		{
			srcidx += (src_jump * inlist[0]);
		}
		else
		{
			size_t destidx = outshape.flat_idx(srccoord);
			memcpy(dest + destidx, src + srcidx, sizeof(T) * basewidth);
			srcidx += inlist[0];
		}
	}
}

template <typename T>
tensor<T>::tensor (void) :
	tensor<T>(std::vector<size_t>{})
{
	set_allocator(default_alloc::alloc_id);
}

template <typename T>
tensor<T>::tensor (T scalar, size_t alloc_id) :
	tensor<T>(std::vector<size_t>{1}, alloc_id)
{
	raw_data_[0] = scalar;
}

template <typename T>
tensor<T>::tensor (tensorshape shape, size_t alloc_id) :
	allowed_shape_(shape)
{
	set_allocator(alloc_id);
	if (allowed_shape_.is_fully_defined())
	{
		alloced_shape_ = shape;
		this->raw_data_ = alloc_->template allocate<T>(
			this->alloced_shape_.n_elems());
	}
}

template <typename T>
tensor<T>::~tensor (void)
{
	alloc_->dealloc(raw_data_, this->alloced_shape_.n_elems());
}

template <typename T>
tensor<T>::tensor (const tensor<T>& other, bool shapeonly)
{
	copy_helper(other, shapeonly);
}

template <typename T>
tensor<T>::tensor (tensor<T>&& other)
{
	move_helper(std::move(other));
}

template <typename T>
tensor<T>& tensor<T>::operator = (const tensor<T>& other)
{
	if (this != &other)
	{
		copy_helper(other, false);
	}
	return *this;
}

template <typename T>
tensor<T>& tensor<T>::operator = (tensor<T>&& other)
{
	if (this != &other)
	{
		move_helper(std::move(other));
	}
	return *this;
}

template <typename T>
tensorshape tensor<T>::get_shape (void) const
{
	if (is_alloc())
	{
		return alloced_shape_;
	}
	return allowed_shape_;
}

template <typename T>
size_t tensor<T>::n_elems (void) const
{
	if (nullptr == raw_data_)
	{
		return 0;
	}
	return this->alloced_shape_.n_elems();
}

template <typename T>
bool tensor<T>::is_alloc (void) const
{
	return alloced_shape_.is_fully_defined() && nullptr != raw_data_;
}

template <typename T>
size_t tensor<T>::total_bytes (void) const
{
	return n_elems() * sizeof(T);
}

// extension of matrix index representation idx = x+y*col
template <typename T>
T tensor<T>::get (std::vector<size_t> coord) const
{
	size_t raw_idx = alloced_shape_.flat_idx(coord);
	if (raw_idx>= alloced_shape_.n_elems())
	{
		throw std::out_of_range(nnutils::formatter() <<
		"out of bound coordinate: " << coord);
	}
	return raw_data_[raw_idx];
}

template <typename T>
std::vector<T> tensor<T>::expose (void) const
{
	assert(is_alloc());
	return std::vector<T>(raw_data_, raw_data_ + n_elems());
}

template <typename T>
void tensor<T>::set_allocator (size_t alloc_id)
{
	if (iallocator* alloc =
		alloc_builder::get_instance().get(alloc_id))
	{
		alloc_ = alloc;
	}
	else
	{
		throw std::runtime_error(nnutils::formatter() << "allocator with id " << alloc_id << " not found");
	}
}

template <typename T>
void tensor<T>::set_shape (tensorshape shape)
{
	// allowed shape update
	if (false == allowed_shape_.is_compatible_with(shape) || false == is_alloc())
	{
		allowed_shape_ = shape;
	}

	// if shape is compatible with alloc then we don't need to change raw data
	// otherwise we need to modify raw data to match new shape
	if (is_alloc() && false == shape.is_compatible_with(alloced_shape_))
	{
		// if shape isn't defined, we need to make it defined
		// by merging with existing allocated shape
		if (false == shape.is_fully_defined())
		{
			// make alloc_shape compatible with shape
			shape = shape.with_rank(alloced_shape_.rank());
			shape = shape.merge_with(alloced_shape_);
			shape = shape.with_rank(allowed_shape_.rank());
		}
		// shape now represent the desired alloced_shape_
		// reshape by allocate
		allocate(shape);
	}
}

template <typename T>
bool tensor<T>::allocate (void)
{
	bool successful = false;
	if (false == is_alloc())
	{
		// alloced_shape_ can be undefined
		if (alloced_shape_.is_fully_defined())
		{
			successful = allocate(alloced_shape_);
		}
		else
		{
			successful = allocate(allowed_shape_);
		}
	}
	return successful;
}

template <typename T>
bool tensor<T>::deallocate (void)
{
	bool success = is_alloc();
	if (success)
	{
		alloc_->dealloc(raw_data_, alloced_shape_.n_elems());
		raw_data_ = nullptr;
		alloced_shape_.undefine();
	}
	return success;
}

template <typename T>
bool tensor<T>::allocate (const tensorshape shape)
{
	bool success = false;
	if (is_alloc() && shape.is_compatible_with(alloced_shape_))
	{
		return success;
	}
	if (shape.is_compatible_with(allowed_shape_) &&
		shape.is_fully_defined())
	{
		success = true;
		// dealloc before reallocation
		if (is_alloc())
		{
			T* temp = alloc_->template allocate<T>(shape.n_elems());
			// move raw_data to temp matching new shape
			// we want to only copy over the minimum data to lower cost
			fit_toshape(temp, shape, raw_data_, alloced_shape_);
			alloc_->dealloc(raw_data_, alloced_shape_.n_elems());
			raw_data_ = temp;
		}
		else
		{
			raw_data_ = alloc_->template allocate<T>(shape.n_elems());
		}
		alloced_shape_ = shape;
	}
	return success;
}

template <typename T>
bool tensor<T>::copy_from (const tensor<T>& other, const tensorshape shape)
{
	bool success = false;
	if (other.is_alloc() &&
		shape.is_fully_defined())
	{
		// allowed shape update
		if (!allowed_shape_.is_compatible_with(shape))
		{
			allowed_shape_ = shape;
		}

		success = true;
		tensorshape olds = other.get_shape();
		T* temp = alloc_->template allocate<T>(shape.n_elems());
		fit_toshape(temp, shape, other.raw_data_, olds);
		
		if (is_alloc())
		{
			alloc_->dealloc(raw_data_, alloced_shape_.n_elems());
		}
		raw_data_ = temp;
		alloced_shape_ = shape;
	}
	return success;
}

// slice along the first dimension
template <typename T>
void tensor<T>::slice (size_t /*dim_start*/, size_t /*limit*/)
{
	throw std::bad_function_call(); // NOT IMPLEMENTED
}

//template <typename T>
// bool shares_buffer_with (const tensor<T>& other) const;

//template <typename T>
// size_t tensor<T>::buffer_hash (void) const {
//	 return 0;
// }

template <typename T>
void tensor<T>::copy_helper (const tensor<T>& other, bool shapeonly)
{
	if (raw_data_)
	{
		alloc_->dealloc(raw_data_, alloced_shape_.n_elems());
		raw_data_ = nullptr;
	}
	alloc_ = other.alloc_;
	alloced_shape_ = other.alloced_shape_;
	allowed_shape_ = other.allowed_shape_;
	if (other.is_alloc())
	{
		size_t ns = alloced_shape_.n_elems();
		raw_data_ = alloc_->template allocate<T>(ns);
		if (false == shapeonly)
		{
			std::memcpy(raw_data_, other.raw_data_, sizeof(T) * ns);
		}
	}
}

template <typename T>
void tensor<T>::move_helper (tensor<T>&& other)
{
	if (raw_data_)
	{
		alloc_->dealloc(raw_data_, alloced_shape_.n_elems());
	}
	// transfer ownership to here.
	raw_data_ = std::move(other.raw_data_);
	// other loses ownership
	other.raw_data_ = nullptr;
	alloc_ = std::move(other.alloc_);
	alloced_shape_ = std::move(other.alloced_shape_);
	allowed_shape_ = std::move(other.allowed_shape_);
}

}

#endif
