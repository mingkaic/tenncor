/*!
 *
 *  agg_func.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph agg_func aggregates input argument 
 *  according to operation opname along specified dimension
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/coord_func.hpp"

#pragma once
#ifndef TENNCOR_AGG_FUNC_HPP
#define TENNCOR_AGG_FUNC_HPP

namespace nnet
{

functor* agg_func (inode* arg, std::string opname, BACKMAP_F bwd);

functor* agg_func (inode* arg, std::string opname, size_t dimension, BACKMAP_F bwd);

}

#endif /* TENNCOR_AGG_FUNC_HPP */
