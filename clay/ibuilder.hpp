/*!
 *
 *  ibuilder.hpp
 *  clay
 *
 *  Purpose:
 *  create a tensor
 *
 *  Created by Mingkai Chen on 2018-05-09.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/tensor.hpp"

#pragma once
#ifndef CLAY_IBUILDER_HPP
#define CLAY_IBUILDER_HPP

namespace clay
{

struct iBuilder
{
	virtual ~iBuilder (void) = default;

	virtual Tensor* get (void) const = 0;

	virtual Tensor* get (Shape shape) const = 0;
};

}

#endif /* CLAY_IBUILDER_HPP */
