/*!
 *
 *  immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph immutable connector that manages a
 *  single operator's forward and backward pass
 *
 *  Created by Mingkai Chen on 2017-02-28.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/base_immutable.hpp"
#include <memory>

#pragma once
#ifndef TENNCOR_IMMUTABLE_HPP
#define TENNCOR_IMMUTABLE_HPP

namespace nnet
{

class immutable : public base_immutable
{
public:
	virtual ~immutable (void);

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for immutables, grabs ownership of Nf
	static immutable* get (std::vector<inode*> args,
		SHAPER shaper, actor_func* Nf,
		BACK_MAP ginit, std::string name,
		inode* ignore_jacobian = nullptr);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	immutable* clone (void) const;

	//! move function
	immutable* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual immutable& operator = (const immutable& other);

	//! declare move assignment to move over transfer functions
	virtual immutable& operator = (immutable&& other);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! immutable constructing an aggregate transfer function
	immutable (std::vector<inode*> args,
		SHAPER shaper, actor_func* Nf,
		BACK_MAP ginit, std::string label);

	//! declare copy constructor to copy over transfer functions
	immutable (const immutable& other);

	//! declare move constructor to move over transfer functions
	immutable (immutable&& other);

	// >>>> POLYMORPHIC CLONERS <<<<
	//! implement clone function
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	// >>>> PROTECTED CLONER <<<<
	//! create a deep copy of this with args
	virtual base_immutable* arg_clone (std::vector<inode*> args) const;

	// >>>> FORWARD & BACKWARD <<<<
	//! forward pass step: populate data_
	virtual void forward_pass (void);

	//! backward pass step: populate gcache_[leaf]
	virtual void backward_pass (variable* leaf);

	itens_actor* actor_ = nullptr;

private:
	//! copy helper
	void copy_helper (const immutable& other);

	//! move helper
	void move_helper (immutable&& other);

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

#endif /* TENNCOR_IMMUTABLE_HPP */
