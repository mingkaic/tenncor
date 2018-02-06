/*!
 *
 *  generator.hpp
 *  cnnet
 *
 *  Purpose:
 *  generate values using init given shape dependency on shape_dep node
 *
 *  Created by Mingkai Chen on 2017-07-18.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/iconnector.hpp"

#pragma once
#ifndef ROCNNET_GENERATOR_HPP
#define ROCNNET_GENERATOR_HPP

namespace nnet
{

class generator : public iconnector
{
public:
	virtual ~generator (void);

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for generator, clones init
	static generator* get (inode* shape_dep,
		const initializer& init, std::string name = "generator");

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	generator* clone (void) const;

	//! move function
	generator* move (void);

	//! declare copy assignment to copy over data and init
	virtual generator& operator = (const generator& other);

	//! declare move assignment to move over data and init
	virtual generator& operator = (generator&& other);

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
	//! check if the arguments are good; data is available
	virtual bool good_status (void) const;

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&);

	// >>>> CALLED BY OBSERVER TO UPDATE <<<<
	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (std::unordered_set<size_t> argidx);

protected:
	// >>>> CONSTRUCTORS <<<<
	//! default constructor
	generator (inode* shape_dep, const initializer& init, std::string name);

	//! declare copy constructor to copy over init and data
	generator (const generator& other);

	//! declare copy constructor to copy over init and data
	generator (generator&& other);

	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone abstraction function
	virtual inode* clone_impl (void) const;

	//! move abstraction function
	virtual inode* move_impl (void);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! get forward passing value
	virtual const tensor* get_eval (void) const;

	//! grab operational gradient node, used by other nodes
	//! adds to internal caches if need be
	virtual inode* get_gradient (variable* leaf);

	// >>>> KILL CONDITION <<<<
	//! suicides when all observers die
	virtual void death_on_broken (void);

	//! suicides when this loses all observers (unless this is_managed)
	virtual void death_on_noparent (void);

private:
	void copy_helper (const generator& other);

	void move_helper (generator&& other);

	void clean_up (void);

	//! initialization handler, owns this
	initializer* init_ = nullptr;

	//! tensor data
	tensor* data_ = nullptr;
};

}

#endif /* ROCNNET_GENERATOR_HPP */
