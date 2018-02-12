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

#include "include/graph/connector/immutable/immutable.hpp"

#pragma once
#ifndef TENNCOR_SHAPE_DEP_HPP
#define TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

// todo: make tensor unaligned
class shape_dep final : public immutable
{
public:
	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for immutables, grabs ownership of Nf
	static shape_dep* get (inode* arg, SHAPE2IDX extracter,
		tensorshape shape, std::string name);

	//! clone function
	shape_dep* clone (void) const;

	//! move function
	shape_dep* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual shape_dep& operator = (const shape_dep& other);

	//! declare move assignment to move over transfer functions
	virtual shape_dep& operator = (shape_dep&& other);

private:
	//! immutable constructing an aggregate transfer function
	shape_dep (inode* arg, SHAPE2IDX extracter,
		tensorshape shape, std::string label);

	//! declare copy constructor to copy over transfer functions
	shape_dep (const shape_dep& other);

	//! declare move constructor to move over transfer functions
	shape_dep (shape_dep&& other);

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


	void copy_helper (const shape_dep& other);

	void move_helper (shape_dep&& other);

	std::shared_ptr<assign_io> asgn_ = std::make_shared<assign_io>();

	//! extract shape dimensions to data_
	SHAPE2IDX extracter_;
};

}

#endif /* TENNCOR_SHAPE_DEP_HPP */
