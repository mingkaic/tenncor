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

#include "include/graph/inode.hpp"

#pragma once
#ifndef TENNCOR_CONSTANT_HPP
#define TENNCOR_CONSTANT_HPP

#include <list>
#include <new>
#include <memory>

namespace nnet
{

class constant final : public inode
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



	// >>>>>>>>>>>> ACCESSORS <<<<<<<<<<<<

	// >>>>>> CONNECTION QUERY <<<<<<

	//! merge/update the gradient/leaf info
	virtual std::unordered_set<const inode*> get_leaves (void) const
	{
		return {this};
	}



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

}

#endif /* TENNCOR_CONSTANT_HPP */
