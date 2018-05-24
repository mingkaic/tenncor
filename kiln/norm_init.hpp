/*!
 *
 *  norm_init.hpp
 *  kiln
 *
 *  Purpose:
 *  built a tensor with a normal distributed values
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "kiln/builder.hpp"

#pragma once
#ifndef KILN_NORM_INIT_HPP
#define KILN_NORM_INIT_HPP

namespace kiln
{

struct NormInit final : public Builder
{
	NormInit (void) = default;

	NormInit (Validator validate);

	NormInit (std::string mean, std::string stdev,
		clay::DTYPE dtype, Validator validate = Validator());

	template <typename T>
	void set (T mean, T stdev)
	{
		dtype_ = clay::get_type<T>();
		if (dtype_ == clay::DTYPE::BAD)
		{
			throw std::exception(); // setting bad type
		}
		mean_ = std::string((char*) &mean, sizeof(T));
		stdev_ = std::string((char*) &stdev, sizeof(T));
	}

protected:
	clay::iBuilder* clone_impl (void) const override
	{
		return new NormInit(*this);
	}

	clay::TensorPtrT build (clay::Shape shape) const override;

private:
	std::string mean_;
	std::string stdev_;
};

}

#endif /* KILN_NORM_INIT_HPP */
