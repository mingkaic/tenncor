/*!
 *
 *  elem_func.hpp
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
#ifndef TENNCOR_ELEM_FUNC_HPP
#define TENNCOR_ELEM_FUNC_HPP

namespace nnet
{

functor* elem_func (std::vector<inode*> args, std::string opname, 
    BACKMAP_F bwd);

}

#endif /* TENNCOR_ELEM_FUNC_HPP */
