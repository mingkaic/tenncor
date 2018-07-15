/*!
 *
 *  shaper.hpp
 *  slip
 *
 *  Purpose:
 *  define common forward shape builders
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "clay/shape.hpp"
#include "mold/inode.hpp"

#ifndef SLIP_SHAPER_HPP
#define SLIP_SHAPER_HPP

namespace slip
{

clay::Shape elem_shape (std::vector<mold::StateRange> states);

clay::Shape relem_shape (std::vector<mold::StateRange> states);

clay::Shape reduce_shape (std::vector<mold::StateRange> states);

clay::Shape matmul_shape (std::vector<mold::StateRange> states);

}

#endif /* SLIP_SHAPER_HPP */
