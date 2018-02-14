/*!
 *
 *  constant.hpp
 *  cnnet
 *
 *  Purpose:
 *  constant node
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
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
	//! builder for scalar
	template <typename T> // todo: optimize by looking for pre-existing constants
	static constant* get (T scalar)
	{
		static_assert(std::is_arithmetic<T>::value, 
			"constant must be arithmetic value");
		tensorshape shape = std::vector<size_t>{1};
		std::shared_ptr<const_init> ci = std::make_shared<const_init>();
		ci->set(scalar);
		return new constant(shape, ci, nnutils::formatter() << scalar);
	}

	//! builder for data and shape
	template <typename T>
	static constant* get (std::vector<T> raw, tensorshape shape)
	{
		static_assert(std::is_arithmetic<T>::value, 
			"constant must be arithmetic value");
		std::string name;
		if (raw.empty())
		{
			name = "<empty>";
		}
		else
		{
			name = nnutils::formatter() << raw.front() << ".." << raw.back();
		}
		std::shared_ptr<const_init> ci = std::make_shared<const_init>();
		ci->set(raw);
		return new constant(shape, ci, name);
	}

	// >>>> CAN'T COPY OR MOVE (GOES AGAINST SHARING) <<<<

	//! deleted copy constructor
	constant (const constant&) = delete;

	//! deleted move constructor
	constant (constant&&) = delete;

	//! copy assignment deleted
	constant& operator = (const constant&) = delete;

	//! move assignment deleted
	constant& operator = (constant&&) = delete;



	// >>>>>>>>>>>> MUTATORS <<<<<<<<<<<<

	//! get tensor data
	virtual tensor* get_tensor (void);

	//! get gradient wrt some node
	virtual varptr derive (inode* wrt);

protected:
	// >>>> KILL CONDITION <<<<
	//! suicides when this loses all observers (unless this is_managed)
	virtual void death_on_noparent (void);

	// >>>> POLYMORPHIC CLONERS (RETURN NULLS) <<<<
	//! clone implementation
	virtual inode* clone_impl (void) const;

	//! move implementation
	virtual inode* move_impl (void);

private:
	//! name constructor, data_ is nullptr
	constant (const tensorshape& shape,
		std::shared_ptr<idata_src> source, std::string name);

	//! raw data
	std::unique_ptr<tensor> data_ = nullptr;
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
