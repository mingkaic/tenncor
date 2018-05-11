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
#ifndef TENSOR_IBUILDER_HPP
#define TENSOR_IBUILDER_HPP

namespace clay
{

struct iBuilder
{
	virtual ~iBuilder (void) = default;

	virtual Tensor* get (void) const;

	virtual Tensor* get (Shape shape) const;
};

}

#endif /* TENSOR_IBUILDER_HPP */
