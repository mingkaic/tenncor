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
#define TENNCOR_SHAPE_DEP_HPP
#ifndef TENNCOR_SHAPE_DEP_HPP
#define TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

using SHAPE_EXTRACT = std::function<std::vector<size_t>(tensorshape&)>;

// todo: make tensor unaligned
class shape_dep : public immutable
{
public:
	virtual ~shape_dep (void);

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for immutables, grabs ownership of Nf
	static shape_dep* get (inode* arg, SHAPE_EXTRACT forward,
		tensorshape shape, std::string name);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	shape_dep* clone (void) const;

	//! move function
	shape_dep* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual shape_dep& operator = (const shape_dep& other);

	//! declare move assignment to move over transfer functions
	virtual shape_dep& operator = (shape_dep&& other);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! immutable constructing an aggregate transfer function
	shape_dep (inode* arg, SHAPE_EXTRACT forward,
		tensorshape shape, std::string label);

	//! declare copy constructor to copy over transfer functions
	shape_dep (const shape_dep& other);

	//! declare move constructor to move over transfer functions
	shape_dep (shape_dep&& other);

	// >>>> POLYMORPHIC CLONERS <<<<
	//! implement clone function
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	// >>>> PROTECTED CLONER <<<<
	//! create a deep copy of this with args
	virtual immutable* arg_clone (std::vector<inode*> args) const;

	// >>>> FORWARD & BACKWARD <<<<
	//! forward pass step: populate data_
	virtual void forward_pass (void);

	//! backward pass step: populate gcache_[leaf]
	virtual void backward_pass (variable* leaf);

private:
	//! extract shape dimensions to data_
	assign_func assigner_;

	SHAPE_EXTRACT extracter_;

	tensorshape shape_;
};

}

#endif /* TENNCOR_SHAPE_DEP_HPP */
#undef TENNCOR_SHAPE_DEP_HPP