/*!
 *
 *  elem_op.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph elem_op connector that manages a
 *  single operator's forward and backward pass
 *
 *  Created by Mingkai Chen on 2017-02-28.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/functor.hpp"

#pragma once
#ifndef TENNCOR_ELEM_OP_HPP
#define TENNCOR_ELEM_OP_HPP

namespace nnet
{

tensorshape elementary_shaper (std::vector<tensorshape> shapes);

functor* reg_func (std::vector<inode*> args, std::string opname, 
    BACKMAP_F bwd, SHAPER_F shaper = elementary_shaper);

}

#endif /* TENNCOR_ELEM_OP_HPP */
