/*!
 *
 *  data_io.hpp
 *  cnnet
 *
 *  Purpose:
 *  tensor object manages shape information and store data
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/tensor/data_src.hpp"

#pragma once
#ifndef TENNCOR_DATA_IO_HPP
#define TENNCOR_DATA_IO_HPP

namespace nnet
{

using SVARR_T = std::pair<std::weak_ptr<void>,tensorshape>;

using GLUE_F = std::function<void(VARR_T,CVAR_T,unsigned short,size_t)>;

using SIDX_F = std::function<std::vector<size_t>(tensorshape,const tensorshape,std::vector<uint64_t>)>;

using OMAP_F = std::function<std::vector<signed>(tensorshape,const tensorshape,std::vector<uint64_t>)>;

struct tens_state final
{
	std::weak_ptr<void> data_;
	tensorshape shape_;
	TENS_TYPE type_;
};

struct idata_dest
{
	virtual ~idata_dest (void) {}

	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx) = 0;
};

struct portal_dest : public idata_dest
{
	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t)
	{
		input_ = {data, shape, type};
	}

	void clear (void)
	{
		input_.data_.reset();
		input_.type_ = BAD_T;
		input_.shape_.undefine();
	}

	tens_state input_;
};

struct assign_io final : virtual idata_src, virtual idata_dest
{
	assign_io (void) {}
	assign_io (const assign_io&) = delete;
	assign_io (assign_io&&) = delete;
	assign_io& operator = (const assign_io&) = delete;
	assign_io& operator = (assign_io&&) = delete;

	assign_io* clone (void) const;
	
	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx);

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

private:
	virtual idata_src* clone_impl (void) const;

	tens_state input_;
};

struct operate_io final : virtual idata_src, virtual idata_dest
{
	// default to homogeneous type
	operate_io (VTFUNC_F op, std::function<TENS_TYPE(std::vector<TENS_TYPE>)> tprocess = 
	[](std::vector<TENS_TYPE> types)
	{
		assert(types.size() > 0 && std::adjacent_find(types.begin(), types.end(), 
			std::not_equal_to<TENS_TYPE>()) == types.end());
		return types[0];
	}) : tprocess_(tprocess), op_(op) {}

	virtual ~operate_io (void) {}

	operate_io* clone (void) const
	{
		return dynamic_cast<operate_io*>(clone_impl());
	}

	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx)
	{
		size_t nargs = args_.size();
		if (idx >= nargs)
		{
			args_.insert(args_.end(), idx - args_.size() + 1, tens_state{});
		}
		args_[idx] = tens_state{data, shape, type};
	}

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

private:
	virtual idata_src* clone_impl (void) const;

	std::function<TENS_TYPE(std::vector<TENS_TYPE>)> tprocess_;

	std::vector<tens_state> args_;

	VTFUNC_F op_;
};

}

#endif /* TENNCOR_DATA_IO_HPP */
