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

#include <memory>

#include "include/tensor/tensorshape.hpp"
#include "include/tensor/type.hpp"
#include "include/operations/data_op.hpp"

#pragma once
#ifndef TENNCOR_DATA_IO_HPP
#define TENNCOR_DATA_IO_HPP

namespace nnet
{

using SHARED_VARR = std::pair<std::shared_ptr<void>,tensorshape>;

using GLUE = std::function<void(SHARED_VARR,const SHARED_VARR,signed short,size_t)>;

struct varr_deleter
{
	void operator () (void* p);
};

std::shared_ptr<void> shared_varr (size_t nbytes);

struct idata_source
{
	virtual ~idata_source (void) {}

	virtual idata_source* clone (void) const = 0;

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape) = 0;
};

struct idata_dest
{
	virtual ~idata_dest (void) {}

	virtual void set_data (std::shared_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx) = 0;
};

struct idata_io : virtual idata_source, virtual idata_dest
{
	virtual ~idata_io (void) {}

	virtual void set_data (std::shared_ptr<void> data, TENS_TYPE type, tensorshape shape, size_t idx)
	{
		assert(type_ == BAD_T || type_ == type); // todo: convert on failure
		type_ = type;
		set_varr(SHARED_VARR{data, shape}, idx);
	}

	virtual void set_varr (SHARED_VARR input, size_t idx) = 0;

protected:
	TENS_TYPE type_;
};

struct const_init final : public idata_source
{
	template <typename T>
	void set (T value)
	{
		type_ = get_type<T>();
		value_ = nnutils::stringify(&value, 1);
	}

	template <typename T>
	void set_vec (std::vector<T> value)
	{
		type_ = get_type<T>();
		value_ = nnutils::stringify(&value[0], value.size());
	}

	virtual idata_source* clone (void) const;

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape);

private:
	std::string value_;

	TENS_TYPE type_;
};

//! Uniformly Random Initialization
struct rand_uniform final : public idata_source
{
	template <typename T>
	void set (T min, T max)
	{
		type_ = get_type<T>();
		min_ = nnutils::stringify(&min, 1);
		max_ = nnutils::stringify(&max, 1);
	}

	virtual idata_source* clone (void) const;

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape);

private:
	std::string min_;
	std::string max_;

	TENS_TYPE type_;
};

//! Normal Random Initialization
struct rand_normal final : public idata_source
{
	template <typename T>
	void set (T mean, T stdev)
	{
		type_ = get_type<T>();
		mean_ = nnutils::stringify(&mean, 1);
		stdev_ = nnutils::stringify(&stdev, 1);
	}

	virtual idata_source* clone (void) const;

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape);

private:
	std::string mean_;
	std::string stdev_;

	TENS_TYPE type_;
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
	TENS_TYPE type_;
	tensorshape shape_;
};



struct open_source final : public idata_source
{
	open_source (std::shared_ptr<idata_source> defsrc);

	virtual idata_source* clone (void) const;

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape);

	std::shared_ptr<idata_source> source_;

private:
	open_source (const open_source& other);
};

struct iholdnrun_io : public idata_io
{
	virtual ~iholdnrun_io (void) {}

	virtual void set_varr (SHARED_VARR input, size_t idx)
	{
		size_t nargs = args_.size();
		if (idx < nargs)
		{
			args_.insert(args_.end(), args_.size() - idx, SHARED_VARR{});
		}
		args_[idx] = input;
	}

protected:
	std::vector<SHARED_VARR> args_;
};

struct operate_io : public iholdnrun_io
{
	operate_io (std::string opname) : opname_(opname) {}

	virtual ~operate_io (void) {}

	virtual idata_source* clone (void) const;

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape);

protected:
	std::string opname_;
};

struct glue_io final : public iholdnrun_io
{
	glue_io (GLUE glue) : glue_(glue) {}

	virtual idata_source* clone (void) const
	{
		return new glue_io(*this);
	}

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape)
	{
		type = type_;
		unsigned short bytesize = type_size(type);
		dest_.first = shared_varr(shape.n_elems() * bytesize);
		dest_.second = shape;
		for (size_t i = 0; i < args_.size(); ++i)
		{
			glue_(dest_, args_[i], bytesize, i);
		}
		return dest_.first;
	}

private:
	SHARED_VARR dest_;

	GLUE glue_;
};

struct assign_io final : public operate_io
{
	assign_io (void) : operate_io("") {}

	assign_io (const assign_io&) = delete;
	assign_io (assign_io&&) = delete;
	assign_io& operator = (const assign_io&) = delete;
	assign_io& operator = (assign_io&&) = delete;

	virtual idata_source* clone (void) const;
	
	void set_op (std::string opname);

	void clear (void);
};

struct sindex_io final : public idata_io
{
	sindex_io (std::vector<size_t> index) : index_(index) {}

	virtual idata_source* clone (void) const;

	virtual void set_varr (SHARED_VARR input, size_t)
	{
		unsigned short bytes = type_size(type_);
		size_t n_elems = input.second.n_elems();
		size_t total_bytes = n_elems * bytes;
		if (nullptr == dest_.first || 
			dest_.second.n_elems() < n_elems)
		{
			dest_.first = shared_varr(total_bytes);
		}
		char* dest = (char*) dest_.first.get();
		char* src = (char*) input.first.get();
		size_t src_idx;
		for (size_t i = 0; i < bytes * index_.size(); ++i)
		{
			src_idx = i / bytes;
			dest[index_[src_idx]] = src[src_idx];
		}
		dest_.second = input.second;
	}

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape)
	{
		assert(type == type_ && shape.is_compatible_with(dest_.second));
		return dest_.first;
	}

private:
	SHARED_VARR dest_;

	std::vector<size_t> index_;
};

}

#endif /* TENNCOR_DATA_IO_HPP */
