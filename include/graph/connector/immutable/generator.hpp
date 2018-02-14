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

#include "include/graph/connector/immutable/immutable.hpp"

#pragma once
#ifndef TENNCOR_GENERATOR_HPP
#define TENNCOR_GENERATOR_HPP

namespace nnet
{

class generator final : public immutable
{
public:
	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for generator, clones init
	static generator* get (inode* shape_dep,
		std::shared_ptr<idata_src> source,
		std::string name = "");

	//! clone function
	generator* clone (void) const;

	//! move function
	generator* move (void);

	//! declare copy assignment to copy over data and init
	virtual generator& operator = (const generator& other);

	//! declare move assignment to move over data and init
	virtual generator& operator = (generator&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	//! get gradient leaves
	virtual std::unordered_set<ileaf*> get_leaves (void) const;

private:
	//! default constructor
	generator (inode* shape_dep, 
		std::shared_ptr<idata_src> source,
		std::string name);

	//! declare copy constructor to copy over init and data
	generator (const generator& other);

	//! declare copy constructor to copy over init and data
	generator (generator&& other);

	// >>>>>> POLYMORPHIC CLONERS <<<<<<

	//! clone abstraction function
	virtual inode* clone_impl (void) const;

	//! move abstraction function
	virtual inode* move_impl (void);

	// >>>>>> FORWARD & BACKWARD <<<<<<

	//! forward pass step: populate data_
	virtual void forward_pass (std::vector<inode*>& args);

	//! backward pass step: populate gcache_[leaf]
	virtual varptr backward_pass (inode* wrt);

	// >>>>>> KILL CONDITION <<<<<<

	//! suicides when this loses all observers
	virtual void death_on_noparent (void);


	void copy_helper (const generator& other);

	void move_helper (generator&& other);

	std::shared_ptr<idata_src> source_;
};

}

#endif /* TENNCOR_GENERATOR_HPP */
