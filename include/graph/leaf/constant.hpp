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
	template <typename T>
	static constant* get (T scalar)
	{
		constant* result = new constant(nnutils::formatter() << scalar);
		result->init(scalar);
		return result;
	}

	//! builder for data and shape
	template <typename T>
	static constant* get (std::vector<T> raw, tensorshape shape)
	{
		std::string name;
		if (raw.empty())
		{
			name = "<empty>";
		}
		else
		{
			name = nnutils::formatter() << raw.front() << ".." << raw.back();
		}
		constant* result = new constant(name);
		result->init(raw, shape);
		return result;
	}

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
	explicit constant (double scalar);

	//! name constructor, data_ is nullptr
	constant (std::string name);

	//! initialize scalar after name constructor
	template <typename T>
	void init (T scalar)
	{
		tenncor::tensor_proto::tensor_t type = get_prototype<T>();
		ileaf::init(std::vector<size_t>{1}, type);

		const_init init;
		init.set(scalar);
		this->data_->allocate(); // ensure allocation
		init(*(this->data_));
		this->is_init_ = true;
	}

	//! initialize raw and shape after name constructor
	template <typename T>
	void init (std::vector<T> raw, tensorshape shape)
	{
		tenncor::tensor_proto::tensor_t type = get_prototype<T>();
		ileaf::init(shape, type);

		size_t rawn = raw.size();
		if (false == this->data_->is_alloc())
		{
			// loosely guess fails if n_elems/n_known> raw size
			// we ensure this will never happen by padding with zeros
			if (shape.n_known()> rawn)
			{
				size_t deficiency = shape.n_known() - rawn;
				raw.insert(raw.end(), deficiency, 0);
			}
			optional<tensorshape> propershape = this->data_->loosely_guess_shape(raw.size());
			assert((bool) propershape);
			this->data_->allocate(*propershape);
		}

		assert(this->data_->is_alloc());
		// we should also pad 0s for well defined shapes
		size_t n = this->data_->n_elems();
		if (n> rawn)
		{
			size_t deficiency = n - rawn;
			raw.insert(raw.end(), deficiency, 0);
		}
		this->assigner_(*(this->data_), (void*) &raw[0], type);
		this->is_init_ = true;
	}

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
	std::vector<T> res = expose<T>(&c);
	return 1 == res.size() && scalar == res[0];
}

//! inequality check for node against scalars
template <typename T>
bool operator != (constant& c, T scalar)
{
	std::vector<T> res = expose<T>(&c);
	return 1 == res.size() && scalar != res[0];
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