/*!
 *
 *  variable.hpp
 *  cnnet
 *
 *  Purpose:
 *  define the graph variable implementation
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/leaf/ivariable.hpp"

#pragma once
#ifndef TENNCOR_VARIABLE_HPP
#define TENNCOR_VARIABLE_HPP

#include <list>
#include <new>
#include <memory>

namespace nnet
{

using variable_updater = std::function<void(bool)>;

class variable final : public ivariable
{
public:
	// >>>> CONSTRUCTORS <<<<
	//! scalar constructor.
	//! all the benefits of constant, but reassignable
	variable (double scalar, std::string name = "scalar");

	//! shape constructor, initializer is null
	variable (const tensorshape& shape, std::string name = "");

	//! shape constructor with initializer
	variable (const tensorshape& shape,
		const initializer<double>& init, std::string name = "");

	// >>>> CLONER <<<<
	//! clone function
	variable* clone (void) const;

	//! move function
	variable* move (void);

	// >>>> VARIABLE SPECIAL <<<<
	//! copy over initializer, replace current initializer
	void set_initializer (const initializer<double>& init);

	//! initialize data and returns if possible,
	//! throws error otherwise
	tensor<double>& initialize (void);

	//! initialize data using shape and
	//! returns if possible, throws error otherwise
	tensor<double>& initialize (tensorshape shape);

	//! return update data function (directly assign input node data to this)
	variable_updater assign (inode* input) const;

	//! return update data function (add input node data to this)
	variable_updater assign_add (inode* input) const;

	//! return update data function (subtract input node data to this)
	variable_updater assign_sub (inode* input) const;

protected:
	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! grab operational gradient node, used by other nodes
	virtual inode* get_gradient (variable* leaf);
};

}

#endif /* TENNCOR_VARIABLE_HPP */
