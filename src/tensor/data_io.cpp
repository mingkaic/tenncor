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

void varr_deleter::operator () (void* p)
{
	free(p);
}

std::shared_ptr<void> shared_varr (size_t nbytes)
{
	return std::shared_ptr<void>(malloc(nbytes), varr_deleter());
}

idata_source* const_init::clone (void) const
{
	return new const_init(*this);
}

std::shared_ptr<void> const_init::get_data (TENS_TYPE& type, tensorshape shape)
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	size_t vbytes = value_.size();
	std::shared_ptr<void> out = shared_varr(nbytes);
	char* dest = (char*) out.get();
	for (size_t i = 0; i < nbytes; i += vbytes)
	{
		memcpy(dest + i, &value_[0], std::min(vbytes, nbytes - i));
	}
	return out;
}

idata_source* rand_uniform::clone (void) const
{
	return new rand_uniform(*this);
}

std::shared_ptr<void> rand_uniform::get_data (TENS_TYPE& type, tensorshape shape)
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	std::shared_ptr<void> out = shared_varr(nbytes);
	tensorshape one(std::vector<size_t>{1});
	operate("rand_uniform", type, VARR{out.get(), shape}, {
		VARR{&min_[0], one},
		VARR{&max_[0], one},
	});
	return out;
}

idata_source* rand_normal::clone (void) const
{
	return new rand_normal(*this);
}

std::shared_ptr<void> rand_normal::get_data (TENS_TYPE& type, tensorshape shape)
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	std::shared_ptr<void> out = shared_varr(nbytes);
	tensorshape one(std::vector<size_t>{1});
	operate("rand_normal", type, VARR{out.get(), shape}, {
		VARR{&mean_[0], one},
		VARR{&stdev_[0], one},
	});
	return out;
}

open_source::open_source (std::shared_ptr<idata_source> defsrc) : source_(defsrc) {}

idata_source* open_source::clone (void) const
{
	return new open_source(*this);
}

std::shared_ptr<void> open_source::get_data (TENS_TYPE& type, tensorshape shape)
{
	assert(nullptr != source_);
	return source_->get_data(type, shape);
}

open_source::open_source (const open_source& other)
{
	source_ = std::shared_ptr<idata_source>(other.source_->clone());
}

idata_source* assign_io::clone (void) const
{
	return new assign_io();
}

void assign_io::set_data (std::shared_ptr<void> data, 
	TENS_TYPE type, tensorshape shape, size_t i)
{
	assert(type_ == BAD_T || type_ == type);
	size_t nargs = args_.size();
	if (i < nargs)
	{
		args_.insert(args_.end(), args_.size() - i, SHARED_VARR{});
	}
	args_[i] = SHARED_VARR{data, shape};
}

std::shared_ptr<void> assign_io::get_data (TENS_TYPE& type, tensorshape shape)
{
	type = type_;
	assert(type != BAD_T && !args_.empty());
	std::shared_ptr<void>& dest = args_[0].first;
	tensorshape& destshape = args_[0].second;
	// todo: check for shapes
	if (!opname_.empty())
	{
		std::vector<VARR> args(args_.size());
		std::transform(args_.begin(), args_.end(), args.begin(), [](SHARED_VARR& sv)
		{
			return VARR{sv.first.get(), sv.second};
		});
		operate(opname_, type_, VARR{dest.get(), destshape}, args);
	}
	return dest;
}

void assign_io::clear (void)
{
	opname_.clear();
	args_.clear();
	type_ = BAD_T;
}

}

#endif