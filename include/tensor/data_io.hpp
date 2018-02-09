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

struct open_source final : public idata_source
{
	open_source (std::shared_ptr<idata_source> defsrc);

	virtual idata_source* clone (void) const;

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape);

	std::shared_ptr<idata_source> source_;

private:
	open_source (const open_source& other);
};

struct assign_io final : virtual idata_source, virtual idata_dest
{
	assign_io (void) {}
	assign_io (const assign_io&) = delete;
	assign_io (assign_io&&) = delete;
	assign_io& operator = (const assign_io&) = delete;
	assign_io& operator = (assign_io&&) = delete;

	virtual idata_source* clone (void) const;

	virtual void set_data (std::shared_ptr<void> data, 
		TENS_TYPE type, tensorshape shape, size_t i);

	virtual std::shared_ptr<void> get_data (TENS_TYPE& type, tensorshape shape);

	void clear (void);

	std::string opname_;
	std::vector<SHARED_VARR> args_;
	TENS_TYPE type_;
};

}

#endif /* TENNCOR_DATA_IO_HPP */
