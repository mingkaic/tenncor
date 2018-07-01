/*!
 *
 *  state_range.hpp
 *  mold
 *
 *  Purpose:
 *  state wrapper containing range information
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/state.hpp"

#include "mold/range.hpp"

#pragma once
#ifndef MOLD_STATE_RANGE_HPP
#define MOLD_STATE_RANGE_HPP

namespace mold
{

struct StateRange
{
	StateRange (clay::State arg, Range drange);

	char* get (void) const;

	clay::Shape shape (void) const;

	clay::DTYPE type (void) const;

	clay::Shape inner (void) const;

	clay::Shape outer (void) const;

	clay::State arg_;

	Range drange_;
};

}

#endif /* MOLD_STATE_RANGE_HPP */
