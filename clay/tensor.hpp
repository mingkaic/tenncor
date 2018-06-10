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

#include "clay/itensor.hpp"

#pragma once
#ifndef CLAY_TENSOR_HPP
#define CLAY_TENSOR_HPP

namespace clay
{

class Tensor final : public iTensor
{
public:
	//! create a tensor of a specified shape
	Tensor (Shape shape, DTYPE dtype);

	Tensor (const Tensor& other);

	Tensor (Tensor&& other) = delete;

	Tensor& operator = (const Tensor& other);

	Tensor& operator = (Tensor&& other) = delete;

	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	//! get internal state
	State get_state (void) const override;

	State get_state (size_t idx) const override;

	//! get tensor shape
	Shape get_shape (void) const override;

	//! get tensor dtype
	DTYPE get_type (void) const override;

	//! get bytes allocated
	size_t total_bytes (void) const override;

private:
	iTensor* clone_impl (void) const override;

	std::shared_ptr<char> data_;

	Shape shape_;

	DTYPE dtype_;
};

}

#endif /* CLAY_TENSOR_HPP */
