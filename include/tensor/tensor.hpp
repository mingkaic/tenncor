/*!
 *
 *  tensor.hpp
 *  cnnet
 *
 *  Purpose:
 *  tensor object manages shape information and store data
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#pragma once
#ifndef TENNCOR_TENSOR_HPP
#define TENNCOR_TENSOR_HPP

#include <stdexcept>
#include <string>
#include <type_traits>
#include <cstring>

#include "include/tensor/type.hpp"
#include "include/tensor/data_io.hpp"

namespace nnet
{

class tensor // todo: make final, and make mock_tensor compose tensor instance
{
public:
	//! create a tensor of a specified shape and allocator
	//! if the shape is fully defined, then raw data is allocated
	//! otherwise, tensor will wait for a defined shape
	tensor (tensorshape shape, std::shared_ptr<idata_source> source); // todo: pass source in as shared

	//! deallocate tensor
	virtual ~tensor (void) {} // remove once final

	//! copy constructor
	tensor (const tensor& other);

	//! move constructor
	tensor (tensor&& other);

	//! copy assignment
	tensor& operator = (const tensor& other);

	//! move assignment
	tensor& operator = (tensor&& other);



	// >>>>>>>>>>>> SERIALIZATION <<<<<<<<<<<<

	//! serialize protobuf tensor
	void serialize (tenncor::tensor_proto* proto_dest) const;

	//! read data and shape from other, take allocator as is
	bool from_proto (const tenncor::tensor_proto& proto_src);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> SHAPE INFORMATION <<<<<<

	//! get tensor shape (allocated if so, allowed shape otherwise)
	tensorshape get_shape (void) const;

	//! get the amount of T elements allocated
	//! if uninitialized, return 0
	size_t n_elems (void) const;

	//! get the tensor rank, number of dimensions
	size_t rank (void) const;

	//! get vector dimension values
	std::vector<size_t> dims (void) const;

	//! checks if input tensor has a compatible allowed tensorshape
	//! or if both this and other are allocated and the trimmed shapes are compatible
	bool is_same_size (const tensor& other) const;

	//! check if other tensor's data is compatible with this shape
	bool is_compatible_with (const tensor& other) const;

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

	// >>>>>> DATA INFORMATION <<<<<<

	void write_to (idata_dest& dest, size_t i = 0) const;

	//! checks if memory is allocated
	bool has_data (void) const;

	//! get bytes allocated
	size_t total_bytes (void) const;
	
	TENS_TYPE get_type (void) const;



	// >>>>>>>>>>>> MUTATOR <<<<<<<<<<<<

	// >>>>>> SHAPE MUTATION <<<<<<

	//! set a new allowed shape
	//! chop raw data outside of new shape
	//! worst case runtime: O(min(N, M))
	//! where N is the original shape size
	//! and M is the resulting shape size
	//! result is shape is compatible with allowed shape
	void set_shape (tensorshape shape);

	// >>>>>> DATA MUTATION <<<<<<

	//! read raw data from source using allowed (innate) shape
	//! return true if successful
	bool read (void);

	//! read raw data from source using input shape
	//! if shape is compatible with allowed
	//! else return false
	bool read (const tensorshape shape);

	//! copy raw data from source using allowed (innate) shape
	//! return true if successful
	//! does not take ownership of array
	bool copy (void);

	//! copy raw data from source using input shape
	//! if shape is compatible with allowed
	//! else return false
	//! does not take ownership of array
	bool copy (const tensorshape shape);

	//! forcefully deallocate raw_data,
	//! invalidates allocated (external) shape
	//! could be useful when we want to preserve allowed shape
	//! since get_shape when allocated gives allocated shape
	bool clear (void);

	//! copy raw_data from other expanded/compressed to input shape
	//! allowed shape will be adjusted similar to set_shape
	bool copy_from (const tensor& other, const tensorshape shape);

	// slice along the first dimension
	void slice (size_t dim_start, size_t limit);

	// >>>>>> DATA EXPOSURE <<<<<<

	std::weak_ptr<idata_source> get_source (void);

private:
	//! copy utility helper
	void copy_helper (const tensor& other);

	//! move utility helper
	void move_helper (tensor&& other);

	// >>>>>> SHAPE MEMBERS <<<<<<

	//! not necessarily defined shape
	tensorshape allowed_shape_;

	//! allocated shape (must be defined)
	tensorshape alloced_shape_;

	// >>>>>> DATA MEMBERS <<<<<<
	std::shared_ptr<idata_source> source_;

	//! raw data is available to tensor manipulators
	std::shared_ptr<void> raw_data_ = nullptr;

	TENS_TYPE dtype_ = BAD_T;
};

}

#endif /* TENNCOR_TENSOR_HPP */
