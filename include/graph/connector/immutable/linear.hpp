/*!
 *
 *  linear.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph linear connector that manages a
 *  single operator's forward and backward pass
 *
 *  Created by Mingkai Chen on 2017-02-28.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/immutable.hpp"
#include <memory>

#pragma once
#ifndef TENNCOR_LINEAR_HPP
#define TENNCOR_LINEAR_HPP

namespace nnet
{

class linear : public immutable
{
public:
	virtual ~linear (void);

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for linear, grabs ownership of Nf
	static linear* get (std::vector<inode*> args,
		SHAPER shaper, actor_func* Nf,
		BACK_MAP ginit, std::string name,
		inode* ignore_jacobian = nullptr);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	linear* clone (void) const;

	//! move function
	linear* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual linear& operator = (const linear& other);

	//! declare move assignment to move over transfer functions
	virtual linear& operator = (linear&& other);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! linear constructing an aggregate transfer function
	linear (std::vector<inode*> args,
		SHAPER shaper, actor_func* Nf,
		BACK_MAP ginit, std::string label);

	//! declare copy constructor to copy over transfer functions
	linear (const linear& other);

	//! declare move constructor to move over transfer functions
	linear (linear&& other);

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

	itens_actor* actor_ = nullptr;

private:
	//! copy helper
	void copy_helper (const linear& other);

	//! move helper
	void move_helper (linear&& other);

	//! calculates shape of this node
	SHAPER shaper_;

	//! forward transfer function
	//! calculates forward passing data
	actor_func* Nf_ = nullptr;

	//! backward transfer function to
	//! lazy instantiate gradient cache values
	BACK_MAP ginit_;
};

}

#endif /* TENNCOR_LINEAR_HPP */
