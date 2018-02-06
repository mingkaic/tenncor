/*!
 *
 *  operations_util.hpp
 *  cnnet
 *
 *  Purpose:
 *  shared utility functions for all operation functions
 *  also useful for debugging purposes
 *
 *  Created by Mingkai Chen on 2017-09-07.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/inode.hpp"

#pragma once
#ifndef TENNCOR_OPERATION_UTILS_HPP
#define TENNCOR_OPERATION_UTILS_HPP

namespace nnet
{

//! return null if no such parent named opname is found, otherwise return parent
inode* unary_parent_search (inode* operand, std::string opname);

//! return null if no such parent satisfies both operands is found, otherwise return parent
inode* ordered_binary_parent_search (inode* a, inode* b, std::string opname);

//! return null if no such parent satisfies both operands is found, otherwise return parent
inode* unordered_binary_parent_search (inode* a, inode* b, std::string opname);

}

#endif /* TENNCOR_OPERATION_UTILS_HPP */
