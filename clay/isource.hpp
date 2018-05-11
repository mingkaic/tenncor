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
#ifndef TENSOR_ISOURCE_HPP
#define TENSOR_ISOURCE_HPP

namespace clay
{

struct iSource
{
	virtual ~iSource (void) = default;

	virtual State get_data (void) const = 0;
};

}

#endif /* TENSOR_ISOURCE_HPP */
