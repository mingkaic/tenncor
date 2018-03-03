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

using SVARR_T = std::pair<std::shared_ptr<void>,tensorshape>;

using GLUE_F = std::function<void(VARR_T,CVAR_T,unsigned short,size_t)>;

using SIDX_F = std::function<std::vector<size_t>(tensorshape,const tensorshape)>;

struct idata_dest
{
	virtual ~idata_dest (void) {}

	virtual void set_data (std::shared_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx) = 0;
};

struct portal_dest : public idata_dest
{
	virtual void set_data (std::shared_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t)
	{
		data_ = data;
		type_ = type;
		shape_ = shape;
	}

	void clear (void)
	{
		data_ = nullptr;
		type_ = BAD_T;
		shape_.undefine();
	}

	std::shared_ptr<void> data_;
	TENS_TYPE type_ = BAD_T;
	tensorshape shape_;
};

struct idata_io : virtual idata_src, virtual idata_dest
{
	virtual ~idata_io (void) {}

	idata_io* clone (void) const
	{
		return dynamic_cast<idata_io*>(this->clone_impl());
	}

	virtual void set_data (std::shared_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx)
	{
		assert(type_ == BAD_T || type_ == type); // todo: convert on failure
		type_ = type;
		set_varr(SVARR_T{data, shape}, idx);
	}

	virtual void set_varr (SVARR_T input, size_t idx) = 0;

protected:
	TENS_TYPE type_ = BAD_T;
};

struct imultiarg_io : public idata_io
{
	virtual ~imultiarg_io (void) {}

	imultiarg_io* clone (void) const
	{
		return dynamic_cast<imultiarg_io*>(this->clone_impl());
	}

	virtual void set_varr (SVARR_T input, size_t idx);

protected:
	std::vector<SVARR_T> args_;
};

struct operate_io : public imultiarg_io
{
	operate_io (std::string opname) : opname_(opname) {}

	virtual ~operate_io (void) {}

	operate_io* clone (void) const
	{
		return dynamic_cast<operate_io*>(clone_impl());
	}

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

protected:
	virtual idata_src* clone_impl (void) const;

	std::string opname_;
};

struct glue_io final : public imultiarg_io
{
	glue_io (GLUE_F glue) : glue_(glue) {}

	glue_io* clone (void) const
	{
		return dynamic_cast<glue_io*>(clone_impl());
	}

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

private:
	virtual idata_src* clone_impl (void) const
	{
		return new glue_io(*this);
	}

	GLUE_F glue_;
};

struct assign_io final : public operate_io
{
	assign_io (void) : operate_io("") {}

	assign_io (const assign_io&) = delete;
	assign_io (assign_io&&) = delete;
	assign_io& operator = (const assign_io&) = delete;
	assign_io& operator = (assign_io&&) = delete;

	assign_io* clone (void) const
	{
		return dynamic_cast<assign_io*>(clone_impl());
	}
	
	void set_op (std::string opname);

	void clear (void);

private:
	virtual idata_src* clone_impl (void) const;
};

struct sindex_io final : public idata_io
{
	sindex_io (SIDX_F smap) : smap_(smap) {}

	sindex_io* clone (void) const
	{
		return dynamic_cast<sindex_io*>(clone_impl());
	}

	virtual void set_varr (SVARR_T input, size_t)
	{
		input_ = input;
	}

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

private:
	virtual idata_src* clone_impl (void) const
	{
		return new sindex_io(*this);
	}

	SIDX_F smap_;

	SVARR_T input_;
	// std::vector<size_t> index_;
};

}

#endif /* TENNCOR_DATA_IO_HPP */
