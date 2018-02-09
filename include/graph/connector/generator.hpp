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
#ifndef TENNCOR_GENERATOR_HPP
#define TENNCOR_GENERATOR_HPP

namespace nnet
{

class generator final : public iconnector
{
public:
	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for generator, clones init
	static generator* get (inode* shape_dep,
		std::shared_ptr<idata_source> source,
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



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr derive (inode* wrt);

	// >>>>>> CALLED BY OBSERVER TO UPDATE <<<<<<

	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (void);

protected:
	//! default constructor
	generator (inode* shape_dep, 
		std::shared_ptr<idata_source> source,
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



	// >>>>>>>>>>>> KILL CONDITION <<<<<<<<<<<<

	//! suicides when all observers die
	virtual void death_on_broken (void);

	//! suicides when this loses all observers (unless this is_managed)
	virtual void death_on_noparent (void);

private:
	void copy_helper (const generator& other);

	void move_helper (generator&& other);

	//! raw data
	std::unique_ptr<tensor> data_ = nullptr;

	std::shared_ptr<idata_source> source_;
};

}

#endif /* TENNCOR_GENERATOR_HPP */
