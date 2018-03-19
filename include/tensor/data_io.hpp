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

struct idata_dest
{
	virtual ~idata_dest (void) {}

	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx) = 0;
};

struct portal_dest : public idata_dest
{
	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t)
	{
		data_ = data;
		type_ = type;
		shape_ = shape;
	}

	void clear (void)
	{
		data_.reset();
		type_ = BAD_T;
		shape_.undefine();
	}

	std::weak_ptr<void> data_;
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

	virtual void set_data (std::weak_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx)
	{
		assert(type_ == BAD_T || type_ == type); // todo: convert on failure
		type_ = type;
		set_varr(SVARR_T{data, shape}, idx);
	}

	virtual void set_varr (SVARR_T input, size_t idx) = 0;

protected:
	TENS_TYPE type_ = BAD_T;
};

struct assign_io final : public idata_io
{
	assign_io (void) {}
	assign_io (const assign_io&) = delete;
	assign_io (assign_io&&) = delete;
	assign_io& operator = (const assign_io&) = delete;
	assign_io& operator = (assign_io&&) = delete;

	assign_io* clone (void) const;

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

	virtual void set_varr (SVARR_T input, size_t);

private:
	virtual idata_src* clone_impl (void) const;

	SVARR_T input_;
};

struct coord_io final : public idata_io
{
	coord_io (OMAP_F inmap) : inmap_(inmap) {}

	virtual ~coord_io (void) {}

	coord_io* clone (void) const
	{
		return dynamic_cast<coord_io*>(clone_impl());
	}

	virtual void set_varr (SVARR_T input, size_t)
	{
		input_ = input;
	}

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
	{
		assert(type_ != BAD_T && !input_.first.expired());
		type = type_;
		size_t per = type_size(type);
		size_t nbytes = shape.n_elems() * per;
		nnutils::check_ptr(outptr, nbytes);
		std::vector<signed> index = inmap_(shape, input_.second, sinfo_);
		char* out = (char*) outptr.get();
		char* in = (char*) input_.first.lock().get();
		for (size_t i = 0; i < index.size(); ++i)
		{
			if (index[i] < 0)
			{
				memset(out + i * per, 0, per);
			}
			else
			{
				memcpy(out + i * per, in + index[i] * per, per);
			}
		}
	}

	virtual void shape_info (std::vector<uint64_t> sinfo)
	{
		sinfo_ = sinfo;
	}

protected:
	virtual idata_src* clone_impl (void) const
	{
		return new coord_io(*this);
	}

	OMAP_F inmap_;

	SVARR_T input_;

	std::vector<uint64_t> sinfo_;
};

struct aggreg_io final : public idata_io
{
	aggreg_io (std::string opname, SIDX_F inmap) : opname_(opname), inmap_(inmap) {}

	virtual ~aggreg_io (void) {}

	aggreg_io* clone (void) const
	{
		return dynamic_cast<aggreg_io*>(clone_impl());
	}

	virtual void set_varr (SVARR_T input, size_t)
	{
		input_ = input;
	}

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const
	{
		assert(type_ != BAD_T && !opname_.empty() && !input_.first.expired());
		type = type_;
		size_t per = type_size(type);
		size_t nbytes = shape.n_elems() * per;
		nnutils::check_ptr(outptr, nbytes);
		std::vector<size_t> index = inmap_(shape, input_.second, sinfo_);
		char* out = (char*) outptr.get();
		char* in = (char*) input_.first.lock().get();
		std::unordered_set<size_t> outmap;
		for (size_t i = 0; i < index.size(); ++i)
		{
			if (outmap.end() == outmap.find(index[i]))
			{
				memcpy(out + index[i] * per, in + i * per, per);
				outmap.insert(index[i]);
			}
			else
			{
				agg_op(opname_, type_, i, out + index[i] * per, in);
			}
		}
	}

	virtual void shape_info (uint64_t dim)
	{
		sinfo_ = {dim};
	}

protected:
	virtual idata_src* clone_impl (void) const
	{
		return new aggreg_io(*this);
	}

	std::string opname_;

	SIDX_F inmap_;

	SVARR_T input_;

	std::vector<uint64_t> sinfo_;
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

	virtual void shape_info (std::vector<uint64_t> info)
	{
		sinfo_ = info;
	}

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

private:
	virtual idata_src* clone_impl (void) const
	{
		return new sindex_io(*this);
	}

	SIDX_F smap_;

	SVARR_T input_;

	std::vector<uint64_t> sinfo_;
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

}

#endif /* TENNCOR_DATA_IO_HPP */
