/*!
 *
 *  isource.hpp
 *  clay
 *
 *  Purpose:
 *  input source
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/state.hpp"

#pragma once
#ifndef CLAY_ISOURCE_HPP
#define CLAY_ISOURCE_HPP

namespace clay
{

struct iSource
{
	virtual ~iSource (void) = default;

	virtual bool read_data (State& dest) const = 0;
};

}

#endif /* CLAY_ISOURCE_HPP */
