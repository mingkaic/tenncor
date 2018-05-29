/*!
 *
 *  validator.hpp
 *  kiln
 *
 *  Purpose:
 *  validate shape and type for tensor build
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <unordered_set>

#include "clay/shape.hpp"
#include "clay/dtype.hpp"

#pragma once
#ifndef KILN_VALIDATOR_HPP
#define KILN_VALIDATOR_HPP

namespace kiln
{

struct EnumHash
{
	template <typename T>
	size_t operator() (T e) const
	{
		return static_cast<size_t>(e);
	}
};

using RejSet = std::unordered_set<clay::DTYPE, EnumHash>;

struct Validator final
{
	Validator (void) = default;

	Validator (clay::Shape allowed, RejSet reject);

	bool support (clay::Shape shape, clay::DTYPE dtype) const;

	clay::Shape allowed_;

	RejSet reject_;
};

}

#endif /* KILN_VALIDATOR_HPP */
