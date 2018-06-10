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

	State (std::weak_ptr<char> data, Shape shape, DTYPE dtype);

	State (char* data, std::weak_ptr<char> block,
		Shape shape, DTYPE dtype);

	State (const State& other, std::weak_ptr<char> block);

	char* get (void) const;

	Shape shape_;
	DTYPE dtype_ = DTYPE::BAD;

private:
	char* data_ = nullptr;

	std::weak_ptr<char> block_;
};

}

#endif /* CLAY_STATE_HPP */
