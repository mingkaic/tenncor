//
//  tensor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include <cstring>

#include "include/tensor/tensor.hpp"

#ifdef TENNCOR_TENSOR_HPP

namespace nnet
{

tensor::tensor (tensorshape shape) :
	allowed_shape_(shape) {}

tensor::tensor (const tenncor::tensor_proto& proto_src)
{
	from_proto(proto_src);
}

tensor::tensor (const tensor& other)
{
	copy_helper(other);
}

tensor::tensor (tensor&& other)
{
	move_helper(std::move(other));
}

tensor& tensor::operator = (const tensor& other)
{
	if (this != &other)
	{
		copy_helper(other);
	}
	return *this;
}

tensor& tensor::operator = (tensor&& other)
{
	if (this != &other)
	{
		move_helper(std::move(other));
	}
	return *this;
}



bool tensor::serialize (tenncor::tensor_proto& proto_dest) const
{
	if (false == has_data()) return false;
	proto_dest.set_type(dtype_);

	// copy bytes
	size_t nb = total_bytes();
	proto_dest.set_data(raw_data_.get(), nb);

	std::vector<size_t> allowed = allowed_shape_.as_list();
	std::vector<size_t> alloced = alloced_shape_.as_list();
	google::protobuf::RepeatedField<uint64_t> allowed_field(allowed.begin(), allowed.end());
	google::protobuf::RepeatedField<uint64_t> alloced_field(alloced.begin(), alloced.end());

	proto_dest.mutable_allowed_shape()->Swap(&allowed_field);
	proto_dest.mutable_alloced_shape()->Swap(&alloced_field);
	return true;
}

void tensor::from_proto (const tenncor::tensor_proto& proto_src)
{
	// shapes must have same dimensionality... (otherwise, input data is definitely corrupt)
	assert(proto_src.alloced_shape_size() == proto_src.allowed_shape_size());
	std::vector<size_t> allowed(proto_src.allowed_shape().begin(), proto_src.allowed_shape().end());
	std::vector<size_t> alloced(proto_src.alloced_shape().begin(), proto_src.alloced_shape().end());
	allowed_shape_ = tensorshape(allowed);
	tensorshape temp_shape(alloced);
	// another sanity check, be less stringent, since this may represent some less evident issue
	assert(temp_shape.is_compatible_with(allowed_shape_) && temp_shape.is_fully_defined());

	clear();
	alloced_shape_ = temp_shape;
	std::string protostr = proto_src.data();

	// copy data over from tensor_proto
	raw_data_ = nnutils::make_svoid(protostr.size());
	memcpy(raw_data_.get(), (void*) protostr.c_str(), protostr.size());

	dtype_ = proto_src.type();
}



tensorshape tensor::get_shape (void) const
{
	if (has_data())
	{
		return alloced_shape_;
	}
	return allowed_shape_;
}

size_t tensor::n_elems (void) const
{
	if (nullptr == raw_data_)
	{
		return 0;
	}
	return this->alloced_shape_.n_elems();
}

size_t tensor::rank (void) const
{
	return get_shape().rank();
}

std::vector<size_t> tensor::dims (void) const
{
	return get_shape().as_list();
}

bool tensor::is_same_size (const tensor& other) const
{
	if (has_data() && other.has_data())
	{
		tensorshape simp_shape = alloced_shape_.trim();
		tensorshape other_simp = other.alloced_shape_.trim();
		return simp_shape.is_compatible_with(other_simp);
	}

	return allowed_shape_.is_compatible_with(other.allowed_shape_);
}

bool tensor::is_compatible_with (const tensor& other) const
{
	return get_shape().is_compatible_with(other.get_shape());
}

bool tensor::is_compatible_with (size_t ndata) const
{
	const tensorshape& my_shape = get_shape();

	bool compatible = true;
	// perfect fit
	if (my_shape.is_fully_defined())
	{
		compatible = ndata == my_shape.n_elems();
	}
	else
	{
		size_t known = my_shape.n_known();
		if (0 < known)
		{
			compatible = 0 == ndata % known;
		}
	}

	return compatible;
}

bool tensor::is_loosely_compatible_with (size_t ndata) const
{
	const tensorshape& my_shape = get_shape();

	bool compatible = true;
	if (my_shape.is_fully_defined())
	{
		compatible = ndata <= my_shape.n_elems();
	}
	// partially defined shapes are always compatible,
	// since unknown dimension can expand infinitely to fit data
	return compatible;
}

optional<tensorshape> tensor::guess_shape (size_t ndata) const
{
	optional<tensorshape> bestshape;
	const tensorshape& allowed_shape = allowed_shape_;
	// if allowed is fully defined
	if (allowed_shape.is_fully_defined())
	{
		if (allowed_shape.n_elems() == ndata)
		{
			bestshape = allowed_shape;
		}
		return bestshape;
	}
	// if allowed is partially defined
	else if (allowed_shape.is_part_defined())
	{
		std::vector<size_t> my_shape = allowed_shape.as_list();
		size_t rank = my_shape.size();
		size_t first_undef = my_shape.size();
		size_t known = 1;
		for (size_t i = 0; i < rank; i++)
		{
			if (0 == my_shape[i])
			{
				if (first_undef> i)
				{
					first_undef = i;
				}
				my_shape[i] = 1;
			}
			else
			{
				known *= my_shape[i];
			}
		}
		assert(known> 0);
		if (0 == ndata % known)
		{
			my_shape[first_undef] = ndata / known;
			bestshape = tensorshape(my_shape);
		}
	}
	// if allowed is undefined
	else
	{
		bestshape = tensorshape({ndata});
	}
	return bestshape;
}

optional<tensorshape> tensor::loosely_guess_shape (size_t ndata) const
{
	if (allowed_shape_.is_fully_defined())
	{
		optional<tensorshape> bestshape;
		if (allowed_shape_.n_elems()>= ndata)
		{
			bestshape = allowed_shape_;
		}
		return bestshape;
	}
	std::vector<size_t> slist = allowed_shape_.as_list();
	size_t first_undef = allowed_shape_.rank();
	size_t known = 1;
	for (size_t i = 0; i < allowed_shape_.rank(); i++)
	{
		if (0 == allowed_shape_[i])
		{
			if (first_undef> i)
			{
				first_undef = i;
			}
			slist[i] = 1;
		}
		else
		{
			known *= allowed_shape_[i];
		}
	}
	slist[first_undef] = ndata / known;
	if (0 != ndata % known)
	{
		// int division above will floor
		// (if we cast to double, we may lose precision)
		slist[first_undef]++;
	}
	return tensorshape(slist);
}

bool tensor::is_aligned (void) const
{
	return true;
}


void tensor::write_to (idata_dest& dest, size_t idx) const
{
	if (false == has_data())
	{
		throw std::exception();
	}
	dest.set_data(raw_data_, dtype_, get_shape(), idx);
}

bool tensor::has_data (void) const
{
	return alloced_shape_.is_fully_defined() && nullptr != raw_data_;
}

size_t tensor::total_bytes (void) const
{
	if (has_data())
	{
		return n_elems() * type_size(dtype_);
	}
	return 0;
}
	
TENS_TYPE tensor::get_type (void) const
{
	return dtype_;
}


void tensor::set_shape (tensorshape shape)
{
	// allowed shape update
	allowed_shape_ = shape;

	// if shape is compatible with alloc then we don't need to change raw data
	// otherwise we need to modify raw data to match new shape
	if (has_data() && false == alloced_shape_.is_compatible_with(shape))
	{
		// shape now represent the desired alloced_shape_
		// reshape by allocate
		clear();
	}
}


bool tensor::read_from (const idata_src& src)
{
	bool successful = nullptr != raw_data_;
	if (successful)
	{
		src.get_data(raw_data_, dtype_, alloced_shape_);
	}
	else if (allowed_shape_.is_fully_defined())
	{
		src.get_data(raw_data_, dtype_, allowed_shape_);
		if (raw_data_ != nullptr)
		{
			alloced_shape_ = allowed_shape_;
		}
		successful = has_data();
	}
	return successful;
}

bool tensor::read_from (const idata_src& src, const tensorshape shape)
{
	bool successful = shape.is_fully_defined() && 
		((nullptr != raw_data_ && 
		shape.is_compatible_with(alloced_shape_)) ||
		shape.is_compatible_with(allowed_shape_));
	if (successful)
	{
		src.get_data(raw_data_, dtype_, shape);
		if (raw_data_ != nullptr)
		{
			alloced_shape_ = shape;
		}
		successful = has_data();
	}
	return successful;
}

bool tensor::clear (void)
{
	bool success = has_data();
	if (success)
	{
		raw_data_ = nullptr;
		alloced_shape_.undefine();
		dtype_ = BAD_T;
	}
	return success;
}

// slice along the first dimension
void tensor::slice (size_t /*dim_start*/, size_t /*limit*/)
{
	throw std::bad_function_call(); // NOT IMPLEMENTED
}


void tensor::copy_helper (const tensor& other)
{
	alloced_shape_ = other.alloced_shape_;
	allowed_shape_ = other.allowed_shape_;
	dtype_ = other.dtype_;

	raw_data_ = nullptr;
	if (other.has_data())
	{
		size_t ns = alloced_shape_.n_elems() * type_size(dtype_);
		raw_data_ = nnutils::make_svoid(ns);
		memcpy(raw_data_.get(), other.raw_data_.get(), ns);
	}
}

void tensor::move_helper (tensor&& other)
{
	// other loses ownership
	alloced_shape_ = std::move(other.alloced_shape_);
	allowed_shape_ = std::move(other.allowed_shape_);
	dtype_ = std::move(other.dtype_);

	raw_data_ = std::move(other.raw_data_);
	other.dtype_ = BAD_T;
}

}

#endif
