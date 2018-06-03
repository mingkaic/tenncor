/*!
 *
 *  const_init.hpp
 *  kiln
 *
 *  Purpose:
 *  built a tensor by a constant value or vector
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/error.hpp"

#include "kiln/builder.hpp"

#pragma once
#ifndef KILN_CONST_INIT_HPP
#define KILN_CONST_INIT_HPP

namespace kiln
{

class ConstInit final : public Builder
{
public:
	ConstInit (void) = default;

	ConstInit (Validator validate);

	ConstInit (std::string data, clay::DTYPE dtype,
		Validator validate = Validator());

	template <typename T>
	void set (T value)
	{
		dtype_ = clay::get_type<T>();
		if (dtype_ == clay::DTYPE::BAD)
		{
			throw clay::UnsupportedTypeError(dtype_);
		}
		data_ = std::string((char*) &value, sizeof(T));
	}

	template <typename T>
	void set (std::vector<T> value)
	{
		dtype_ = clay::get_type<T>();
		if (dtype_ == clay::DTYPE::BAD)
		{
			throw clay::UnsupportedTypeError(dtype_);
		}
		data_ = std::string((char*) &value[0], sizeof(T) * value.size());
	}

protected:
	clay::iBuilder* clone_impl (void) const override
	{
		return new ConstInit(*this);
	}

	clay::TensorPtrT build (clay::Shape shape) const override;

private:
	std::string data_;
};

}

#endif /* KILN_CONST_INIT_HPP */
