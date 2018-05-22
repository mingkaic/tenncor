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

	iBuilder* clone (void) const
	{
		return clone_impl();
	}

	virtual TensorPtrT get (void) const = 0;

	virtual TensorPtrT get (Shape shape) const = 0;

protected:
	virtual iBuilder* clone_impl (void) const = 0;
};

}

#endif /* CLAY_IBUILDER_HPP */
