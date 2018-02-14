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

static void fit_toshape (size_t bytesize, char* dest, const tensorshape& outshape, const char* src, const tensorshape& inshape)
{
	assert(outshape.is_fully_defined() && inshape.is_fully_defined());
	std::vector<size_t> outlist = outshape.as_list();
	std::vector<size_t> inlist = inshape.as_list();
	size_t numout = outshape.n_elems();

	size_t minrank = std::min(outlist.size(), inlist.size());
	tensorshape clippedshape = inshape.with_rank(minrank);
	size_t numin = clippedshape.n_elems();

	memset(dest, 0, bytesize * numout);
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
			memcpy(dest + destidx, src + srcidx, bytesize * basewidth);
			srcidx += inlist[0];
		}
	}
}

tensor::tensor (tensorshape shape) :
	allowed_shape_(shape) {}

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



void tensor::serialize (tenncor::tensor_proto* proto_dest) const
{
	if (false == has_data()) return;
	proto_dest->set_type(dtype_);

	// copy bytes
	size_t nb = total_bytes();
	proto_dest->set_data(raw_data_.get(), nb);

	std::vector<size_t> allowed = allowed_shape_.as_list();
	std::vector<size_t> alloced = alloced_shape_.as_list();
	google::protobuf::RepeatedField<uint64_t> allowed_field(allowed.begin(), allowed.end());
	google::protobuf::RepeatedField<uint64_t> alloced_field(alloced.begin(), alloced.end());

	proto_dest->mutable_allowed_shape()->Swap(&allowed_field);
	proto_dest->mutable_alloced_shape()->Swap(&alloced_field);
}

bool tensor::from_proto (const tenncor::tensor_proto& proto_src)
{
	// shapes must have same dimensionality... (otherwise, input data is definitely corrupt)
	assert(proto_src.alloced_shape_size() == proto_src.allowed_shape_size());
	std::vector<size_t> allowed(proto_src.allowed_shape().begin(), proto_src.allowed_shape().end());
	std::vector<size_t> alloced(proto_src.alloced_shape().begin(), proto_src.alloced_shape().end());
	allowed_shape_ = tensorshape(allowed);
	tensorshape temp_shape(alloced);
	// another sanity check, be less stringent, since this may represent some less evident issue
	if (false == temp_shape.is_compatible_with(allowed_shape_) ||
		false == temp_shape.is_fully_defined()) return false;

	clear();
	alloced_shape_ = temp_shape;
	std::string protostr = proto_src.data();

	// copy data over from tensor_proto
	raw_data_ = shared_varr(protostr.size());
	memcpy(raw_data_.get(), (void*) protostr.c_str(), protostr.size());

	dtype_ = proto_src.type();
	return true;
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
	std::vector<size_t> my_shape = allowed_shape_.as_list();
	size_t first_undef = my_shape.size();
	size_t known = 1;
	for (size_t i = 0; i < my_shape.size(); i++)
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
	my_shape[first_undef] = ndata / known;
	if (0 != ndata % known)
	{
		// int division above will floor
		// (if we cast to double, we may lose precision)
		my_shape[first_undef]++;
	}
	return tensorshape(my_shape);
}

bool tensor::is_aligned (void) const
{
	return true;
}


void tensor::write_to (idata_dest& dest, size_t idx) const
{
	dest.set_data(raw_data_, dtype_, get_shape(), idx);
}

bool tensor::has_data (void) const
{
	return alloced_shape_.is_fully_defined() && nullptr != raw_data_;
}

size_t tensor::total_bytes (void) const
{
	return n_elems() * type_size(dtype_);
}
	
TENS_TYPE tensor::get_type (void) const
{
	return dtype_;
}


void tensor::set_shape (tensorshape shape)
{
	// allowed shape update
	if (false == allowed_shape_.is_compatible_with(shape) || nullptr == raw_data_)
	{
		allowed_shape_ = shape;
	}

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
	// assert that alloced_shape is undefined if not allocated
	if (nullptr == raw_data_ && allowed_shape_.is_fully_defined())
	{
		src.get_data(raw_data_, dtype_, allowed_shape_);
		if (raw_data_ != nullptr)
		{
			alloced_shape_ = allowed_shape_;
		}
	}
	return has_data();
}

bool tensor::read_from (const idata_src& src, const tensorshape shape)
{
	if (nullptr == raw_data_ &&
		shape.is_compatible_with(allowed_shape_)&&
		shape.is_fully_defined())
	{
		src.get_data(raw_data_, dtype_, shape);
		if (raw_data_ != nullptr)
		{
			alloced_shape_ = shape;
		}
	}
	return has_data();
}

bool tensor::clear (void)
{
	bool success = has_data();
	if (success)
	{
		raw_data_ = nullptr;
		alloced_shape_.undefine();
	}
	return success;
}

bool tensor::copy_from (const tensor& other, const tensorshape shape)
{
	bool success = other.has_data() && shape.is_fully_defined();
	if (success)
	{
		// allowed shape update
		if (!allowed_shape_.is_compatible_with(shape))
		{
			allowed_shape_ = shape;
		}

		tensorshape olds = other.get_shape();
		dtype_ = other.dtype_;
		size_t bsize = type_size(dtype_);

		raw_data_ = shared_varr(bsize * shape.n_elems());
		fit_toshape(bsize, (char*) raw_data_.get(), shape, (char*) other.raw_data_.get(), olds);

		alloced_shape_ = shape;
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
	raw_data_ = nullptr;
	if (other.has_data())
	{
		size_t ns = alloced_shape_.n_elems();
		raw_data_ = shared_varr(ns);
		memcpy(raw_data_.get(), other.raw_data_.get(), ns);
	}

	alloced_shape_ = other.alloced_shape_;
	allowed_shape_ = other.allowed_shape_;
}

void tensor::move_helper (tensor&& other)
{
	raw_data_ = std::move(other.raw_data_);

	// other loses ownership
	alloced_shape_ = std::move(other.alloced_shape_);
	allowed_shape_ = std::move(other.allowed_shape_);
}

}

#endif
