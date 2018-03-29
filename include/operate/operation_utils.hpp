/*!
 *
 *  operation_utils.hpp
 *  cnnet
 *
 *  Purpose:
 *  find argument parents to prevent 
 *  excessive duplicate operations
 *
 *  Created by Mingkai Chen on 2017-09-07.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/functor.hpp"

#ifndef TENNCOR_OPERATION_UTILS_HPP
#define TENNCOR_OPERATION_UTILS_HPP

namespace nnet
{

inode* single_parent (inode* src, OPCODE opcode);

inode* ordered_parent (std::vector<inode*> srcs, OPCODE opcode);

inode* unordered_parent (std::vector<inode*> srcs, OPCODE opcode);

}

#endif /* TENNCOR_OPERATION_UTILS_HPP */
