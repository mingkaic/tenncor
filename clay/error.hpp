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

struct NilDataError : public std::runtime_error
{
	NilDataError (void);
};

struct UnsupportedTypeError : public std::runtime_error
{
	UnsupportedTypeError (clay::DTYPE type);
};

struct InvalidShapeError : public std::runtime_error
{
	InvalidShapeError (clay::Shape shape);

	InvalidShapeError (clay::Shape shape, clay::Shape shape2);
};

}

#endif /* CLAY_ERROR_HPP */
