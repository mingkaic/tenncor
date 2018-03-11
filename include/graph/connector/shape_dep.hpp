/*!
 *
 *  shape_dep.hpp
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

#include "include/graph/connector/functor.hpp"

#pragma once
#ifndef TENNCOR_SHAPE_DEP_HPP
#define TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

using USIDX_F = std::function<std::vector<size_t>(tensorshape)>;

functor* shape_func (inode* arg, USIDX_F extracter, USHAPE_F shaper, std::string label);

}

#endif /* TENNCOR_SHAPE_DEP_HPP */
