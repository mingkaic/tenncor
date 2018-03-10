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
#ifndef TENNCOR_IMMUTABLE_HPP
#define TENNCOR_IMMUTABLE_HPP

namespace nnet
{

class immutable : public iconnector
{
public:
	virtual ~immutable (void);

	//! clone function
	immutable* clone (void) const;

	//! move function
	immutable* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual immutable& operator = (const immutable& other);

	//! declare move assignment to move over transfer functions
	virtual immutable& operator = (immutable&& other);



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<
	
	//! get gradient leaves
	virtual std::unordered_set<inode*> get_leaves (void) const;



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr derive (inode* wrt);

	// >>>>>> CALLED BY OBSERVER TO UPDATE <<<<<<

	//! Inherited from iobserver: update data
	virtual void update (void);

protected:
	//! immutable constructing an aggregate transfer function
	immutable (std::vector<inode*> args, std::string label);

	//! declare copy constructor to copy over transfer functions
	immutable (const immutable& other);

	//! declare move constructor to move over transfer functions
	immutable (immutable&& other);



	// >>>>>>>>>>>> FORWARD & BACKWARD <<<<<<<<<<<<

	//! forward pass step: populate data_
	virtual void forward_pass (std::vector<inode*>& args) = 0;

	virtual varptr backward_pass (inode* wrt) = 0;

	// todo: have an option to disable data_ caching for performance boost
	//! inner tensor to cache forward evaluated values
	std::unique_ptr<tensor> data_ = nullptr;

private:
	//! copy helper
	void copy_helper (const immutable& other);

	//! move helper
	void move_helper (immutable&& other);
};

}

#endif /* TENNCOR_IMMUTABLE_HPP */
