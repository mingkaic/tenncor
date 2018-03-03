/*!
 *
 *  data_src.hpp
 *  cnnet
 *
 *  Purpose:
 *  input source into tensor
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <memory>

#include "include/utils/shared_ptr.hpp"
#include "include/tensor/tensorshape.hpp"
#include "include/tensor/type.hpp"
#include "include/operations/data_op.hpp"

#pragma once
#ifndef TENNCOR_DATA_SRC_HPP
#define TENNCOR_DATA_SRC_HPP

namespace nnet
{

struct idata_src
{
	virtual ~idata_src (void);

	idata_src* clone (void) const;

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const = 0;

protected:
	virtual idata_src* clone_impl (void) const = 0;
};

struct const_init final : public idata_src
{
	template <typename T>
	void set (T value)
	{
		type_ = get_type<T>();
		value_ = nnutils::stringify(&value, 1);
	}

	template <typename T>
	void set (std::vector<T> value)
	{
		type_ = get_type<T>();
		value_ = nnutils::stringify(&value[0], value.size());
	}

	const_init* clone (void) const;

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

private:
	virtual idata_src* clone_impl (void) const;

	std::string value_;

	TENS_TYPE type_ = BAD_T;
};

//! Uniformly Random Initialization
struct rand_uniform final : public idata_src
{
	template <typename T>
	void set (T min, T max)
	{
		type_ = get_type<T>();
		min_ = nnutils::stringify(&min, 1);
		max_ = nnutils::stringify(&max, 1);
	}

	rand_uniform* clone (void) const;

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

private:
	virtual idata_src* clone_impl (void) const;

	std::string min_;

	std::string max_;

	TENS_TYPE type_ = BAD_T;
};

//! Normal Random Initialization
struct rand_normal final : public idata_src
{
	template <typename T>
	void set (T mean, T stdev)
	{
		type_ = get_type<T>();
		mean_ = nnutils::stringify(&mean, 1);
		stdev_ = nnutils::stringify(&stdev, 1);
	}

	rand_normal* clone (void) const;

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

private:
	virtual idata_src* clone_impl (void) const;

	std::string mean_;
	std::string stdev_;

	TENS_TYPE type_ = BAD_T;
};

}

#endif /* TENNCOR_DATA_SRC_HPP */
