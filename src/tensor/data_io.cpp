//
//	data_io.cpp
//	cnnet
//
//	Created by Mingkai Chen on 2018-01-12.
//	Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/tensor/data_io.hpp"

#ifdef TENNCOR_DATA_IO_HPP

namespace nnet
{

assign_io* assign_io::clone (void) const
{
	return dynamic_cast<assign_io*>(clone_impl());
}

void assign_io::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	assert(type_ != BAD_T && !input_.first.expired() &&
		shape.is_compatible_with(input_.second));
	type = type_;
	// assert shape.is_compatible_with(input_.second);
	size_t nbytes = shape.n_elems() * type_size(type);
	nnutils::check_ptr(outptr, nbytes);
	memcpy(outptr.get(), input_.first.lock().get(), nbytes);
}

void assign_io::set_varr (SVARR_T input, size_t)
{
	input_ = input;
}

idata_src* assign_io::clone_impl (void) const
{
	return new assign_io();
}


void sindex_io::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	std::vector<size_t> index = smap_(shape, input_.second);
	type = this->type_;
	// implicity assert(shape.n_elems() <= *std::max_element(index_.begin(), index_.end()))
	unsigned short bytes = type_size(type);
	nnutils::check_ptr(outptr, index.size() * bytes);
	char* dest = (char*) outptr.get();
	assert(!input_.first.expired());
	const char* src = (const char*) input_.first.lock().get();
	for (size_t i = 0; i < index.size(); ++i)
	{
		std::memcpy(dest + i * bytes, src + index[i] * bytes, bytes);
	}
}


void imultiarg_io::set_varr (SVARR_T input, size_t idx)
{
	size_t nargs = args_.size();
	if (idx >= nargs)
	{
		args_.insert(args_.end(), idx - args_.size() + 1, SVARR_T{});
	}
	args_[idx] = input;
}


void operate_io::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	assert(!opname_.empty());
	type = type_;
	assert(type != BAD_T && !args_.empty());
	size_t nbytes = shape.n_elems() * type_size(type);
	nnutils::check_ptr(outptr, nbytes);
	// todo: check for shapes
	std::vector<CVAR_T> args(args_.size());
	std::transform(args_.begin(), args_.end(), args.begin(), [](const SVARR_T& sv)
	{
		assert(!sv.first.expired());
		return CVAR_T{sv.first.lock().get(), sv.second};
	});
	ele_op(opname_, type_, VARR_T{outptr.get(), shape}, args);
}

idata_src* operate_io::clone_impl (void) const
{
	return new operate_io(*this);
}


void glue_io::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const 
{ 
	type = type_; 
	unsigned short bytesize = type_size(type); 
	size_t nbytes = shape.n_elems() * bytesize; 
	nnutils::check_ptr(outptr, nbytes); 
	void* dest = outptr.get(); 
	std::memset(dest, 0, nbytes); 
	for (size_t i = 0; i < args_.size(); ++i) 
	{ 
		assert(!args_[i].first.expired()); 
		glue_(VARR_T{dest, shape}, CVAR_T{args_[i].first.lock().get(), args_[i].second}, bytesize, i); 
	} 
}

}

#endif
