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

using TYPE_F = std::function<TENS_TYPE(std::vector<TENS_TYPE>)>;

using GLUE_F = std::function<void(VARR_T,CVAR_T,unsigned short,size_t)>;

using SIDX_F = std::function<std::vector<size_t>(tshape,const tshape,std::vector<uint64_t>)>;

using OMAP_F = std::function<std::vector<signed>(tshape,const tshape,std::vector<uint64_t>)>;

struct tens_state final
{
	tens_state (void) : type_(BAD_T) {}

	tens_state (std::weak_ptr<void> data,
		tshape shape, TENS_TYPE type) : 
		data_(data), shape_(shape), type_(type) {}

	tens_state (const tens_state& other) :
		data_(other.data_),
		shape_(other.shape_),
		type_(other.type_) {}
	
	tens_state (tens_state&& other) :
		data_(std::move(other.data_)),
		shape_(std::move(other.shape_)),
		type_(other.type_) { other.type_ = BAD_T; }

	tens_state& operator = (const tens_state& other)
	{
		if (this != &other)
		{
			data_ = other.data_;
			shape_ = other.shape_;
			type_ = other.type_; 
		}
		return *this;
	}

	tens_state& operator = (tens_state&& other)
	{
		if (this != &other)
		{
			data_ = std::move(other.data_);
			shape_ = std::move(other.shape_);
			type_ = other.type_;
			other.type_ = BAD_T;
		}
		return *this;
	}

	std::weak_ptr<void> data_;
	tshape shape_;
	TENS_TYPE type_;
};

struct idata_dest
{
	virtual ~idata_dest (void) = default;

	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tshape shape, size_t idx) = 0;
};

struct portal_dest : public idata_dest
{
	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tshape shape, size_t);

	void clear (void);

	tens_state input_;
};

struct assign_io final : virtual idata_src, virtual idata_dest
{
	assign_io (void) {}

	virtual ~assign_io (void) {}

	assign_io* clone (void) const;

	assign_io (assign_io&& other) :
		idata_src(std::move(other)), idata_dest(std::move(other)),
		input_(std::move(other.input_)) {}
	
	assign_io& operator = (const assign_io& other)
	{
		if (this != &other)
		{
			input_ = other.input_;
		}
		return *this;
	}

	assign_io& operator = (assign_io&& other)
	{
		if (this != &other)
		{
			input_ = std::move(other.input_);
		}
		return *this;
	}
	
	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tshape shape, size_t idx);

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tshape shape) const;

private:
	assign_io (const assign_io& other) : 
		idata_src(other), idata_dest(other),
		input_(other.input_) {}

	virtual idata_src* clone_impl (void) const;

	tens_state input_;
};

struct operate_io final : virtual idata_src, virtual idata_dest
{
	// default to homogeneous type
	operate_io (VTFUNC_F op, TYPE_F tprocess = 
	[](std::vector<TENS_TYPE> types)
	{
		assert(types.size() > 0 && std::adjacent_find(types.begin(), types.end(), 
			std::not_equal_to<TENS_TYPE>()) == types.end());
		return types[0];
	}) : tprocess_(tprocess), op_(op) {}

	virtual ~operate_io (void) {}

	operate_io* clone (void) const;

	operate_io (operate_io&& other) :
		idata_src(std::move(other)), idata_dest(std::move(other)),
		tprocess_(std::move(other.tprocess_)),
		args_(std::move(other.args_)), op_(std::move(other.op_)) {}
	
	operate_io& operator = (const operate_io& other)
	{
		if (this != &other)
		{
			tprocess_ = other.tprocess_;
			args_ = other.args_;
			op_ = other.op_;
		}
		return *this;
	}

	operate_io& operator = (operate_io&& other)
	{
		if (this != &other)
		{
			tprocess_ = std::move(other.tprocess_);
			args_ = std::move(other.args_);
			op_ = std::move(other.op_);
		}
		return *this;
	}

	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tshape shape, size_t idx);

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tshape shape) const;

private:
	operate_io (const operate_io& other) :
		idata_src(other), idata_dest(other),
		tprocess_(other.tprocess_),
		args_(other.args_), op_(other.op_) {}

	virtual idata_src* clone_impl (void) const;

	std::function<TENS_TYPE(std::vector<TENS_TYPE>)> tprocess_;

	std::vector<tens_state> args_;

	VTFUNC_F op_;
};

}

#endif /* TENNCOR_DATA_IO_HPP */
