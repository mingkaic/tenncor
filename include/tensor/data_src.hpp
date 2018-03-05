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
#include "include/utils/error.hpp"

#pragma once
#ifndef TENNCOR_DATA_SRC_HPP
#define TENNCOR_DATA_SRC_HPP

namespace nnet
{

using GENERIC = std::pair<std::string, TENS_TYPE>;

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
		if (type_ == BAD_T)
		{
			throw std::exception(); // setting bad type
		}
		value_ = nnutils::stringify(&value, 1);
	}

	template <typename T>
	void set (std::vector<T> value)
	{
		type_ = get_type<T>();
		if (type_ == BAD_T)
		{
			throw std::exception(); // setting bad type
		}
		value_ = nnutils::stringify(&value[0], value.size());
	}

	const_init* clone (void) const;

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

	GENERIC get_const (void) const
	{
		return {value_, type_};
	}

private:
	virtual idata_src* clone_impl (void) const;

	std::string value_;

	TENS_TYPE type_ = BAD_T;
};

//! Uniformly Random Initialization
struct r_uniform_init final : public idata_src
{
	template <typename T>
	void set (T min, T max)
	{
		type_ = get_type<T>();
		if (type_ == BAD_T)
		{
			throw std::exception(); // setting bad type
		}
		min_ = nnutils::stringify(&min, 1);
		max_ = nnutils::stringify(&max, 1);
	}

	r_uniform_init* clone (void) const;

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

	GENERIC get_min (void) const
	{
		return {min_, type_};
	}

	GENERIC get_max (void) const
	{
		return {max_, type_};
	}

private:
	virtual idata_src* clone_impl (void) const;

	std::string min_;

	std::string max_;

	TENS_TYPE type_ = BAD_T;
};

//! Normal Random Initialization
struct r_normal_init final : public idata_src
{
	template <typename T>
	void set (T mean, T stdev)
	{
		type_ = get_type<T>();
		if (type_ == BAD_T)
		{
			throw std::exception(); // setting bad type
		}
		mean_ = nnutils::stringify(&mean, 1);
		stdev_ = nnutils::stringify(&stdev, 1);
	}

	r_normal_init* clone (void) const;

	virtual void get_data (std::shared_ptr<void>& outptr, TENS_TYPE& type, tensorshape shape) const;

	GENERIC get_mean (void) const
	{
		return {mean_, type_};
	}

	GENERIC get_stdev (void) const
	{
		return {stdev_, type_};
	}

private:
	virtual idata_src* clone_impl (void) const;

	std::string mean_;
	std::string stdev_;

	TENS_TYPE type_ = BAD_T;
};

}

#endif /* TENNCOR_DATA_SRC_HPP */
