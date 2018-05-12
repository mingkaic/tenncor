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

struct Validator final
{
	Validator (void) = default;

	Validator (clay::Shape allowed, std::unordered_set<clay::DTYPE> reject);

	bool support (clay::Shape shape, clay::DTYPE dtype) const;

	clay::Shape allowed_;

	std::unordered_set<clay::DTYPE> reject_;
};

}

#endif /* KILN_VALIDATOR_HPP */
