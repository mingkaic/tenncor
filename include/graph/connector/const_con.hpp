/*!
 *
 *  const_con.hpp
 *  cnnet
 *
 *  Purpose:
 *  an connector extension that
 *  clears all leaves and mimic constant behavior
 *
 *  Created by Mingkai Chen on 2017-09-17.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/connector/immutable/immutable.hpp"

#pragma once
#ifndef TENNCOR_CONST_CON_HPP
#define TENNCOR_CONST_CON_HPP

namespace nnet
{

class const_con final : public iconnector
{
public:
	static const_con* get (inode* x);

	//! clone function
	const_con* clone (void) const;

	//! move function
	const_con* move (void);

	//! declare copy assignment to copy over transfer functions
	virtual const_con& operator = (const const_con& other);

	//! declare move assignment to move over transfer functions
	virtual const_con& operator = (const_con&& other);



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

private:
	const_con (inode* x);

	//! declare copy constructor to copy over transfer functions
	const_con (const const_con& other);

	//! declare move constructor to move over transfer functions
	const_con (const_con&& other);

	// >>>>>> POLYMORPHIC CLONERS <<<<<<

	virtual inode* clone_impl (void) const;

	virtual inode* move_impl (void);
};

}

#endif /* TENNCOR_CONST_CON_HPP */
