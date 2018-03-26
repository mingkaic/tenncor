/*!
 *
 *  coord_func.hpp
 *  cnnet
 *
 *  Purpose:
 *  maps source data 1 to 1 with dest data via index map
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/func/functor.hpp"

#pragma once
#ifndef TENNCOR_COORD_FUNC_HPP
#define TENNCOR_COORD_FUNC_HPP

namespace nnet
{

functor* coord_func (std::vector<inode*> args, VTFUNC_F cf, USHAPE_F shaper, OPCODE op);

}

#endif /* TENNCOR_COORD_FUNC_HPP */
