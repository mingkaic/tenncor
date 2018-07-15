/*!
 *
 *  range.hpp
 *  mold
 *
 *  Purpose:
 *  range representation and apply to shape
 *  to separate inner and outer shape
 *
 *  Created by Mingkai Chen on 2018-06-20
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/shape.hpp"

#pragma once
#ifndef MOLD_RANGE_HPP
#define MOLD_RANGE_HPP

namespace mold
{

struct Range
{
	Range (size_t lower, size_t upper);

	clay::Shape apply (const clay::Shape& inshape) const;

	clay::Shape front (const clay::Shape& inshape) const;

	clay::Shape back (const clay::Shape& inshape) const;

	size_t lower_;

	size_t upper_;
};

}

#endif /* MOLD_RANGE_HPP */
