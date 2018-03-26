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

void portal_dest::set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t)
{
	input_ = {data, shape, type};
}

void portal_dest::clear (void)
{
	input_.data_.reset();
	input_.type_ = BAD_T;
	input_.shape_.undefine();
}



assign_io* assign_io::clone (void) const
{
	return dynamic_cast<assign_io*>(clone_impl());
}

void assign_io::set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx)
{
	input_ = {data, shape, type};
}

void assign_io::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	assert(input_.type_ != BAD_T && !input_.data_.expired() &&
		shape.is_compatible_with(input_.shape_));
	type = input_.type_;
	// assert shape.is_compatible_with(input_.second);
	size_t nbytes = shape.n_elems() * type_size(type);
	nnutils::check_ptr(outptr, nbytes);
	memcpy(outptr.get(), input_.data_.lock().get(), nbytes);
}

idata_src* assign_io::clone_impl (void) const
{
	return new assign_io();
}



operate_io* operate_io::clone (void) const
{
	return dynamic_cast<operate_io*>(clone_impl());
}

void operate_io::set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx)
{
	size_t nargs = args_.size();
	if (idx >= nargs)
	{
		args_.insert(args_.end(), idx - args_.size() + 1, tens_state{});
	}
	args_[idx] = tens_state{data, shape, type};
}

void operate_io::get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
{
	std::vector<TENS_TYPE> argtypes(args_.size());
	std::transform(args_.begin(), args_.end(), argtypes.begin(),
	[](const tens_state& state)
	{
		return state.type_;
	});
	TENS_TYPE outtype = tprocess_(argtypes);
	assert(op_ && outtype != BAD_T && !args_.empty());
	type = outtype;
	size_t nbytes = shape.n_elems() * type_size(type);
	nnutils::check_ptr(outptr, nbytes);
	// todo: check for shapes
	std::vector<CVAR_T> cargs(args_.size());
	std::transform(args_.begin(), args_.end(), cargs.begin(),
	[](const tens_state& state)
	{
		assert(!state.data_.expired());
		return CVAR_T{state.data_.lock().get(), state.shape_};
	});
	op_(outtype, VARR_T{outptr.get(), shape}, cargs);
}

idata_src* operate_io::clone_impl (void) const
{
	return new operate_io(*this);
}

}

#endif
