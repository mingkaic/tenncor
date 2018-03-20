/*!
 *
 *  shape_func.hpp
 *  cnnet
 *
 *  Purpose:
 *  Obtain shape information from dependencies
 *  Shape is organized by:
 *  	Each row i represents the info of dependency i
 *  	Each column element represents the dimensional value
 *  Tensor data_ is guaranteed to be 2-D
 *
 *  Created by Mingkai Chen on 2017-07-03.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/func/functor.hpp"

#pragma once
#ifndef TENNCOR_SHAPE_DEP_HPP
#define TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

using USIDX_F = std::function<std::vector<size_t>(tensorshape, std::vector<uint64_t>)>;

functor* shape_func (std::vector<inode*> args, USIDX_F extracter, USHAPE_F shaper, OPCODE op);

}

#endif /* TENNCOR_SHAPE_DEP_HPP */
