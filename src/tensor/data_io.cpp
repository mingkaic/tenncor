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

static inline void check_ptr (std::shared_ptr<void>& ptr, size_t nbytes)
{
	if (nullptr == ptr)
	{
		ptr = shared_varr(nbytes);
	}
}

struct varr_deleter
{
	void operator () (void* p)
	{
		free(p);
	}
};

std::shared_ptr<void> shared_varr (size_t nbytes)
{
	return std::shared_ptr<void>(malloc(nbytes), varr_deleter());
}


idata_src* const_init::clone_impl (void) const
{
	return new const_init(*this);
}

void const_init::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	size_t vbytes = value_.size();
	check_ptr(outptr, nbytes);
	char* dest = (char*) outptr.get();
	for (size_t i = 0; i < nbytes; i += vbytes)
	{
		memcpy(dest + i, &value_[0], std::min(vbytes, nbytes - i));
	}
}


idata_src* rand_uniform::clone_impl (void) const
{
	return new rand_uniform(*this);
}

void rand_uniform::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	check_ptr(outptr, nbytes);
	tensorshape one(std::vector<size_t>{1});
	operate("rand_uniform", type, VARR_T{outptr.get(), shape}, {
		CVAR_T{&min_[0], one},
		CVAR_T{&max_[0], one},
	});
}


idata_src* rand_normal::clone_impl (void) const
{
	return new rand_normal(*this);
}

void rand_normal::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	check_ptr(outptr, nbytes);
	tensorshape one(std::vector<size_t>{1});
	operate("rand_normal", type, VARR_T{outptr.get(), shape}, {
		CVAR_T{&mean_[0], one},
		CVAR_T{&stdev_[0], one},
	});
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
	check_ptr(outptr, nbytes);
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
	check_ptr(outptr, shape.n_elems() * bytesize);
	for (size_t i = 0; i < args_.size(); ++i)
	{
		glue_(VARR_T{outptr.get(), shape}, CVAR_T{args_[i].first.get(), args_[i].second}, bytesize, i);
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
	type = this->type_;
	// implicity assert(shape.n_elems() <= *std::max_element(index_.begin(), index_.end()))
	unsigned short bytes = type_size(type);
	size_t n_elems = shape.n_elems();
	check_ptr(outptr, n_elems * bytes);
	char* dest = (char*) outptr.get();
	char* src = (char*) input_.get();
	size_t src_idx;
	for (size_t i = 0; i < bytes * index_.size(); ++i)
	{
		src_idx = i / bytes;
		dest[index_[src_idx]] = src[src_idx];
	}
}

}

#endif
