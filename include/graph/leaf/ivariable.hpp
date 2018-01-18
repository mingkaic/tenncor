/*!
 *
 *  ivariable.hpp
 *  cnnet
 *
 *  Purpose:
 *  variable interface
 *
 *  Created by Mingkai Chen on 2017-02-27.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/leaf/constant.hpp"

#pragma once
#ifndef TENNCOR_IVARIABLE_HPP
#define TENNCOR_IVARIABLE_HPP

namespace nnet
{

class ivariable : public ileaf
{
public:
	virtual ~ivariable (void);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	ivariable* clone (void) const;

	//! move function
	ivariable* move (void);

	//! declare copy assignment to copy over initializer
	virtual ivariable& operator = (const ivariable& other);

	//! declare move assignment to move over initializer
	virtual ivariable& operator = (ivariable&& other);

	// >>>> BACKWARD DATA <<<<
	//! get gradient wrt some node
	virtual varptr derive (inode* wrt);

	// >>>> IVARIABLE SPECIAL <<<<
	//! determine whether leaf node can be initiated
	bool can_init (void) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! construct to init zero and one
	ivariable (const tensorshape& shape,
		tenncor::tensor_proto::tensor_t type,
		initializer* init, std::string name);

	//! copy construct to init zero and one
	ivariable (const ivariable& other);

	//! move construct to init zero and one
	ivariable (ivariable&& other);

	//! initialization handler, owns this
	initializer* init_ = nullptr;

private:
	//! copy helper
	void copy_helper (const ivariable& other);

	//! move helper
	void move_helper (ivariable&& other);
};

}

#endif /* TENNCOR_IVARIABLE_HPP */
