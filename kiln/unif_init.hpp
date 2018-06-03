/*!
 *
 *  unif_init.hpp
 *  kiln
 *
 *  Purpose:
 *  built a tensor with a uniform distributed values
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/error.hpp"

#include "kiln/builder.hpp"

#pragma once
#ifndef KILN_UNIF_INIT_HPP
#define KILN_UNIF_INIT_HPP

namespace kiln
{

struct UnifInit final : public Builder
{
	UnifInit (void) = default;

	UnifInit (Validator validate);

	UnifInit (std::string min, std::string max,
		clay::DTYPE dtype, Validator validate = Validator());

	template <typename T>
	void set (T min, T max)
	{
		dtype_ = clay::get_type<T>();
		if (dtype_ == clay::DTYPE::BAD)
		{
			throw clay::UnsupportedTypeError(dtype_);
		}
		min_ = std::string((char*) &min, sizeof(T));
		max_ = std::string((char*) &max, sizeof(T));
	}

protected:
	clay::iBuilder* clone_impl (void) const override
	{
		return new UnifInit(*this);
	}

	clay::TensorPtrT build (clay::Shape shape) const override;

private:
	std::string min_;

	std::string max_;
};

}

#endif /* KILN_UNIF_INIT_HPP */
