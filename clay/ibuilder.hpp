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

using TensorPtrT = std::unique_ptr<Tensor>;

struct iBuilder
{
	virtual ~iBuilder (void) = default;

	virtual TensorPtrT get (void) const = 0;

	virtual TensorPtrT get (Shape shape) const = 0;
};

}

#endif /* CLAY_IBUILDER_HPP */
