//
//  data_io.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/tensor/data_io.hpp"

#ifdef TENNCOR_DATA_IO_HPP

namespace nnet
{

void imultiarg_io::set_varr (SVARR_T input, size_t idx)
{
	size_t nargs = args_.size();
	if (idx >= nargs)
	{
		args_.insert(args_.end(), idx - args_.size() + 1, SVARR_T{});
	}
	args_[idx] = input;
}


idata_src* operate_io::clone_impl (void) const
{
	return new operate_io(*this);
}

void operate_io::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	type = type_;
	assert(type != BAD_T && !args_.empty());
	size_t nbytes = shape.n_elems() * type_size(type);
	nnutils::check_ptr(outptr, nbytes);
	// todo: check for shapes
	if (!opname_.empty())
	{
		std::vector<CVAR_T> args(args_.size());
		std::transform(args_.begin(), args_.end(), args.begin(), [](const SVARR_T& sv)
		{
			return CVAR_T{sv.first.get(), sv.second};
		});
		operate(opname_, type_, VARR_T{outptr.get(), shape}, args);
	}
	else // identity operation (copy over)
	{
		memcpy(outptr.get(), args_[0].first.get(), nbytes);
	}
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
		glue_(VARR_T{dest, shape}, CVAR_T{args_[i].first.get(), args_[i].second}, bytesize, i);
	}
}


idata_src* assign_io::clone_impl (void) const
{
	return new assign_io();
}

void assign_io::set_op (std::string opname)
{
	opname_ = opname;
}

void assign_io::clear (void)
{
	opname_.clear();
	args_.clear();
	type_ = BAD_T;
}


void sindex_io::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	std::vector<size_t> index = smap_(shape, input_.second);
	type = this->type_;
	// implicity assert(shape.n_elems() <= *std::max_element(index_.begin(), index_.end()))
	unsigned short bytes = type_size(type);
	nnutils::check_ptr(outptr, index.size() * bytes);
	char* dest = (char*) outptr.get();
	const char* src = (const char*) input_.first.get();
	for (size_t i = 0; i < index.size(); ++i)
	{
		std::memcpy(dest + i * bytes, src + index[i] * bytes, bytes);
	}
}

}

#endif
