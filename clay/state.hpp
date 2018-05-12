/*!
 *
 *  state.hpp
 *  clay
 *
 *  Purpose:
 *  state is a transactional structure for tensor changes
 *
 *  Created by Mingkai Chen on 2018-05-09.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <memory>

#include "clay/dtype.hpp"
#include "clay/shape.hpp"

#pragma once
#ifndef CLAY_STATE_HPP
#define CLAY_STATE_HPP

namespace clay
{

struct State final
{
	State (void) = default;

	State (std::weak_ptr<const char> data, Shape shape, DTYPE dtype);

	std::weak_ptr<const char> data_;
	Shape shape_;
	DTYPE dtype_ = DTYPE::BAD;
};

}

#endif /* CLAY_STATE_HPP */
