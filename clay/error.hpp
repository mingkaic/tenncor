/*!
 *
 *  error.hpp
 *  clay
 *
 *  Purpose:
 *  model common tensor error cases
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <stdexcept>

#include "clay/shape.hpp"
#include "clay/dtype.hpp"

#pragma once
#ifndef CLAY_ERROR_HPP
#define CLAY_ERROR_HPP

namespace clay
{

struct UnsupportedTypeError : public std::runtime_error
{
	UnsupportedTypeError (DTYPE type);
};

struct InvalidShapeError : public std::runtime_error
{
	InvalidShapeError (Shape shape);

	InvalidShapeError (Shape shape, Shape shape2);
};

}

#endif /* CLAY_ERROR_HPP */
