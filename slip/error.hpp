/*!
 *
 *  error.hpp
 *  slip
 *
 *  Purpose:
 *  model common operational cases
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <stdexcept>

#include "slip/opcode.hpp"

#include "clay/shape.hpp"
#include "clay/dtype.hpp"

#pragma once
#ifndef SLIP_ERROR_HPP
#define SLIP_ERROR_HPP

namespace slip
{

struct NoArgumentsError : public std::runtime_error
{
	NoArgumentsError (void);
};

struct BadNArgsError : public std::runtime_error
{
	BadNArgsError (size_t nexpect, size_t ngot);
};

struct UnsupportedOpcodeError : public std::runtime_error
{
	UnsupportedOpcodeError (OPCODE opcode);
};

struct ShapeMismatchError : public std::runtime_error
{
	ShapeMismatchError (clay::Shape shape, clay::Shape other);
};

struct TypeMismatchError : public std::runtime_error
{
	TypeMismatchError (clay::DTYPE type, clay::DTYPE other);
};

struct InvalidDimensionError : public std::runtime_error
{
	InvalidDimensionError (uint64_t dim, clay::Shape shape);
};

struct InvalidRangeError : public std::runtime_error
{
	InvalidRangeError (mold::Range range, clay::Shape shape);
};

}

#endif /* SLIP_ERROR_HPP */
