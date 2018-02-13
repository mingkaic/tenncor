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

#include "include/graph/connector/immutable/immutable.hpp"

#pragma once
#ifndef TENNCOR_ELEM_OP_HPP
#define TENNCOR_ELEM_OP_HPP

namespace nnet
{

using FWD_MAP = std::function<tensor*(std::vector<const tensor*>)>;

class elem_op final : public immutable
{
public:
	// >>>>>>>>>>>> BUILDER TO FORCE HEAP ALLOCATION <<<<<<<<<<<<

	//! builder for elem_op, grabs ownership of Nf
	static elem_op* get (std::vector<inode*> args, std::string opname, BACK_MAP bwd);

	//! build with a definitive outshape
	static elem_op* get (std::vector<inode*> args, tensorshape shape, std::string opname, BACK_MAP bwd);

	//! clone function
	elem_op* clone (void) const;

	//! move function
	elem_op* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual elem_op& operator = (const elem_op& other);

	//! declare move assignment to move over transfer functions
	virtual elem_op& operator = (elem_op&& other);

private:
	//! elem_op constructing an aggregate transfer function
	elem_op (std::vector<inode*> args, std::string opname, BACK_MAP bwd);

	//! create elem_op with definiteive outshape
	elem_op (std::vector<inode*> args, tensorshape shape, std::string opname, BACK_MAP bwd);

	//! declare copy constructor to copy over transfer functions
	elem_op (const elem_op& other);

	//! declare move constructor to move over transfer functions
	elem_op (elem_op&& other);

	// >>>>>> POLYMORPHIC CLONERS <<<<<<

	//! implement clone function
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	// >>>>>> FORWARD & BACKWARD <<<<<<

	//! forward pass step: populate data_
	virtual void forward_pass (std::vector<inode*>& args);

	//! backward pass step
	virtual varptr backward_pass (inode* wrt);


	//! copy helper
	void copy_helper (const elem_op& other);

	//! move helper
	void move_helper (elem_op&& other);

	//! calculates shape of this node
	SHAPER shaper_;
	
	FWD_MAP fwd_;

	//! backward transfer function to
	//! lazy instantiate gradient cache values
	BACK_MAP bwd_;
};

}

#endif /* TENNCOR_ELEM_OP_HPP */

