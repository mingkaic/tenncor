/*!
 *
 *  isource.hpp
 *  mold
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
#ifndef MOLD_ISOURCE_HPP
#define MOLD_ISOURCE_HPP

namespace mold
{

struct iSource
{
	virtual ~iSource (void) = default;

	virtual bool write_data (clay::State& dest) const = 0;
};

}

#endif /* MOLD_ISOURCE_HPP */
