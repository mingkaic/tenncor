/*!
 *
 *  common.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph common connectors that manages a forward and backward pass
 *
 *  Created by Mingkai Chen on 2017-02-28.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/functor.hpp"

#pragma once
#ifndef TENNCOR_COM_FUNC_HPP
#define TENNCOR_COM_FUNC_HPP

namespace nnet
{

using USIDX_F = std::function<std::vector<size_t>(tensorshape, std::vector<uint64_t>)>;

functor* elem_func (std::vector<inode*> args, std::string opname, 
	OPCODE op, BACKMAP_F bwd);

functor* elem_func (std::vector<inode*> args, std::string opname, 
	OPCODE op, BACKMAP_F bwd, TYPE_F tprocess);

functor* coord_func (std::vector<inode*> args, VTFUNC_F cf, USHAPE_F shaper, OPCODE op);

functor* agg_func (inode* arg, std::string opname, OPCODE op, BACKMAP_F bwd);

functor* agg_func (inode* arg, inode* dimension, std::string opname, OPCODE op, BACKMAP_F bwd);

functor* shape_func (std::vector<inode*> args, USIDX_F extracter, USHAPE_F shaper, OPCODE op);

}

#endif /* TENNCOR_COM_FUNC_HPP */
