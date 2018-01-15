/*!
 *
 *  constant.hpp
 *  cnnet
 *
 *  Purpose:
 *  constant node
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "include/graph/leaf/ileaf.hpp"

#pragma once
#ifndef TENNCOR_CONSTANT_HPP
#define TENNCOR_CONSTANT_HPP

#include <list>
#include <new>
#include <memory>

namespace nnet
{

class constant final : public ileaf
{
public:
	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! get shared zero constant that is managed
	static constant* get_shared_zero (void);
	
	//! get shared one constant that is managed
	static constant* get_shared_one (void);
	
	//! builder for scalar
	static constant* get (double scalar);

	//! builder for data and shape
	static constant* get (std::vector<double> raw, tensorshape shape);

	// >>>> CAN'T COPY OR MOVE (GOES AGAINST SHARING) <<<<
	//! deleted copy constructor
	constant (const constant& other) = delete;

	//! deleted move constructor
	constant (constant&& other) = delete;

	//! copy assignment deleted
	constant& operator = (const constant& other) = delete;

	//! move assignment deleted
	constant& operator = (constant&& other) = delete;

	// >>>> BACKWARD DATA <<<<
	//! get gradient wrt some node
	virtual varptr derive (inode* wrt);

	// >>>> NODE STATUS <<<<
	//! set this constant as being managed by some node
	//! this will not die if it loses all observers
	void be_managed (void);

protected:
	//! scalar constructor
	constant (double scalar);

	//! raw and shape constructor
	constant (std::vector<double>raw, tensorshape shape);

	// >>>> KILL CONDITION <<<<
	//! suicides when this loses all observers (unless this is_managed)
	virtual void death_on_noparent (void);

	// >>>> POLYMORPHIC CLONERS (RETURN NULLS) <<<<
	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

	// >>>> INTERNAL DATA TRANSFERS <<<<
	//! grab operational gradient node, used by other nodes
	virtual inode* get_gradient (variable* leaf);

private:
	//! if constant is managed by some node,
	//! that node is responsible for this node's life cycle
	bool is_managed_ = false;
};

//! equality check for node against scalars
template <typename T>
bool operator == (constant& c, T scalar)
{
	std::vector<T>res = expose<T>(&c);
	return 1 == res.size() && scalar == res[0];
}

//! inequality check for node against scalars
template <typename T>
bool operator != (constant& c, T scalar)
{
	std::vector<T>res = expose<T>(&c);
	return 1 != res.size() || scalar != res[0];
}

//! create a constant with zeros everywhere except for all elements with index
//! at a specified dimension where these elements are filled with scalar
//! const_axis(2, I, S, {...}) => constant[:, I, :, ...] = S
template <typename T>
constant* const_axis (size_t dimension, size_t index, T scalar, tensorshape shape)
{
	std::vector<double>data(shape.n_elems(), 0);
	shape.iterate([&data, dimension, index, scalar](std::vector<size_t> coord, size_t idx)
	{
		if (coord[dimension] == index)
		{
			data[idx] = scalar;
		}
	});
	return constant::get(data, shape);
}

}

#endif /* TENNCOR_CONSTANT_HPP */