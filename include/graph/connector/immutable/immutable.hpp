/*!
 *
 *  immutable.hpp
 *  cnnet
 *
 *  Purpose:
 *  graph immutable connector interface
 *  manages tensor data and defines abstract
 *  forward computation and backward computation methods
 *
 *  also defines mergable immutable for a
 *  series of forward and backward passes
 *
 *  Created by Mingkai Chen on 2017-06-26.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/iconnector.hpp"
#include "include/graph/leaf/variable.hpp"

#pragma once
#define TENNCOR_IMMUTABLE_HPP
#ifndef TENNCOR_IMMUTABLE_HPP
#define TENNCOR_IMMUTABLE_HPP

namespace nnet
{

class immutable : public iconnector
{
public:
	//! type for mapping leaf nodes to derivative with respect to leaf
	using GRAD_CACHE = std::unordered_map<ileaf*,varptr>;

	virtual ~immutable (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	immutable* clone (void) const;

	//! move function
	immutable* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual immutable& operator = (const immutable& other);

	//! declare move assignment to move over transfer functions
	virtual immutable& operator = (immutable&& other);

	// >>>> FORWARD & BACKWARD DATA <<<<
	//! grab a temporary value traversing top-down
	//! allocates out tensor. caller owns out
	virtual void temporary_eval (const iconnector* target, inode*& out) const;

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr derive (inode* wrt);

	//! Utility function: get data shape
	virtual tensorshape get_shape (void) const;

	// >>>> GRAPH STATUS <<<<
	//! get gradient leaves
	virtual std::unordered_set<ileaf*> get_leaves (void) const;

	// >>>> NODE STATUS <<<<

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&);

	// >>>> CALLED BY OBSERVER TO UPDATE <<<<
	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (std::unordered_set<size_t> argidx);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! immutable constructing an aggregate transfer function
	immutable (std::vector<inode*> args, std::string label);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! declare copy constructor to copy over transfer functions
	immutable (const immutable& other);

	//! declare move constructor to move over transfer functions
	immutable (immutable&& other);

	// >>>> PROTECTED CLONER <<<<
	//! create a deep copy of this with args
	virtual immutable* arg_clone (std::vector<inode*> args) const = 0;

	// >>>> KILL CONDITION <<<<
	//! suicides when all observers die
	void death_on_broken (void);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! Forward passing value
	virtual const tensor* get_eval (void) const;

	//! grab operational gradient node, used by other nodes
	//! delay instantiate gcache elements if target leaf was never instantiated
	virtual inode* get_gradient (variable* leaf);

	// >>>> FORWARD & BACKWARD <<<<
	//! forward pass step: populate data_
	virtual void forward_pass (void) = 0;

	//! backward pass step: populate gcache_[leaf]
	virtual void backward_pass (variable* leaf) = 0;

	//! maps leaf to gradient node
	//! lazy instantiates gradient nodes
	//! - stores the gradient value wrt each leaf
	//! - record leaf set
	typename immutable::GRAD_CACHE gcache_;

	// todo: have an option to disable data_ caching for performance boost
	//! inner tensor to cache forward evaluated values
	tensor* data_ = nullptr;

private:
	//! copy helper
	void copy_helper (const immutable& other);

	//! move helper
	void move_helper (immutable&& other);

	//! temporary_eval helper
	inode* temp_eval_helper (const iconnector* target, constant*& out) const;
};

}

#endif /* TENNCOR_IMMUTABLE_HPP */
#undef TENNCOR_IMMUTABLE_HPP