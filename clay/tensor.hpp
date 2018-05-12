/*!
 *
 *  tensor.hpp
 *  clay
 *
 *  Purpose:
 *  tensor manages data, shape, and type
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <stdexcept>
#include <string>
#include <type_traits>
#include <cstring>

#include "clay/shape.hpp"
#include "clay/dtype.hpp"

#include "clay/state.hpp"
#include "clay/isource.hpp"

#pragma once
#ifndef CLAY_CLAY_HPP
#define CLAY_CLAY_HPP

namespace clay
{

class Tensor final
{
public:
	//! create a tensor of a specified shape
	Tensor (std::shared_ptr<char> data, Shape shape, DTYPE dtype);

	//! other.dtype_ is BAD afterwards
	Tensor (Tensor&& other);

	//! other.dtype_ is BAD afterwards
	Tensor& operator = (Tensor&& other);

	// >>>> AVOID OVERRIDE <<<<
	Tensor (const Tensor&) = default;

	Tensor& operator = (const Tensor&) = default;


	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	//! get internal state
	State get_state (void) const;

	//! get tensor shape
	Shape get_shape (void) const;

	//! get tensor dtype
	DTYPE get_type (void) const;

	//! get bytes allocated
	size_t total_bytes (void) const;


	// >>>>>>>>>>>> MUTATOR <<<<<<<<<<<<

	//! copy over data from src
	//! return true if successful
	bool read_from (const iSource& src);

private:
	std::shared_ptr<char> data_;

	Shape shape_;

	DTYPE dtype_;
};

}

#endif /* CLAY_CLAY_HPP */
