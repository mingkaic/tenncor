//
//  data_src.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2018-01-12.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/tensor/data_src.hpp"

#ifdef TENNCOR_DATA_SRC_HPP

namespace nnet
{

idata_src::~idata_src (void) {}

idata_src* idata_src::clone (void) const
{
	return this->clone_impl();
}


struct const_init* const_init::clone (void) const
{
	return static_cast<const_init*>(clone_impl());
}

void const_init::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	size_t vbytes = value_.size();
	nnutils::check_ptr(outptr, nbytes);
	char* dest = (char*) outptr.get();
	for (size_t i = 0; i < nbytes; i += vbytes)
	{
		memcpy(dest + i, &value_[0], std::min(vbytes, nbytes - i));
	}
}

idata_src* const_init::clone_impl (void) const
{
	return new const_init(*this);
}


struct r_uniform_init* r_uniform_init::clone (void) const
{
	return static_cast<r_uniform_init*>(clone_impl());
}

void r_uniform_init::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	nnutils::check_ptr(outptr, nbytes);
	tensorshape one(std::vector<size_t>{1});
	ebind("rand_uniform")(type, VARR_T{outptr.get(), shape}, {
		CVAR_T{&min_[0], one},
		CVAR_T{&max_[0], one},
	});
}

idata_src* r_uniform_init::clone_impl (void) const
{
	return new r_uniform_init(*this);
}


struct r_normal_init* r_normal_init::clone (void) const
{
	return static_cast<r_normal_init*>(clone_impl());
}

void r_normal_init::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	type = type_;
	size_t nbytes = shape.n_elems() * type_size(type);
	nnutils::check_ptr(outptr, nbytes);
	tensorshape one(std::vector<size_t>{1});
	ebind("rand_normal")(type, VARR_T{outptr.get(), shape}, {
		CVAR_T{&mean_[0], one},
		CVAR_T{&stdev_[0], one},
	});
}

idata_src* r_normal_init::clone_impl (void) const
{
	return new r_normal_init(*this);
}

}

#endif
