/*!
 *
 *  placeholder.hpp
 *  cnnet
 *
 *  Purpose:
 *  placeholder implementation
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include <list>
#include <new>
#include <memory>

#include "include/graph/leaf/ivariable.hpp"

#pragma once
#ifndef TENNCOR_PLACEHOLDER_HPP
#define TENNCOR_PLACEHOLDER_HPP

namespace nnet
{

class placeholder final : public ivariable
{
public:
	// >>>> CONSTRUCTORS <<<<
	//! shape constructor
	placeholder (const tensorshape& shape, std::string name = "");

	//! explicitly declare copy constructor since assignments are declared
	placeholder (const placeholder& other);

	//! explicitly declare move constructor since assignments are declared
	placeholder (placeholder&& other);

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	placeholder* clone (void) const;

	//! move function
	placeholder* move (void);

	//! declare copy assignment to avoid implicit deletion
	virtual placeholder& operator = (const placeholder& other);

	//! declare move assignment to avoid implicit deletion
	virtual placeholder& operator = (placeholder&& other);

	// >>>> DATA ASSIGNMENT OPERATORS <<<<
	//! assign raw data according to a
	//! vector representation of inner tensor
	//! for a shape of <d_0, d_1, ..., d_i> and
	//! 	coordinate <c_0, c_1, ..., c_i>:
	//! index mapping function is
	//! sum_j=0:i(product_k=0:j(d_k-1) * c_j) where for k < 0 d_k = 1
	virtual placeholder& operator = (std::vector<double>data);

	//! assign tensor to inner tensor
	virtual placeholder& operator = (tensor<double>& data);

protected:
	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! grab operational gradient node, used by other nodes
	virtual inode* get_gradient (variable* );
};

class placeptr : public varptr
{
public:
	//! nullptr construction
	placeptr (void) {}

	//! wrap placeholder pointer
	placeptr (placeholder* ptr);

	//! assign a pointer
	placeptr& operator = (placeholder* other);

	// >>>> EXTENDING PLACEHOLDER <<<<
	//! assign a raw data
	placeptr& operator = (std::vector<double>vec);

	//! assign a tensor
	placeptr& operator = (tensor<double>& ten);

	//! implicit pointer conversion
	operator placeholder* () const;

	//! dereference overload
	placeholder& operator * (void);

	//! pointer accessor overload
	placeholder* operator -> (void);

	//! get inner pointer as placeholder pointer
	placeholder* get (void) const;
};

}

#endif /* TENNCOR_PLACEHOLDER_HPP */
