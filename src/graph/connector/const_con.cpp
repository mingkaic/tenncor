//
//  const_con.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-17.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/const_con.hpp"

#ifdef TENNCOR_CONST_CON_HPP

namespace nnet
{

const_con* const_con::get (inode* x)
{
	return new const_con(x);
}

const_con* const_con::clone (void) const
{
	return static_cast<const_con*>(clone_impl());
}

const_con* const_con::move (void)
{
	return static_cast<const_con*>(move_impl());
}

const_con& const_con::operator = (const const_con& other)
{
	if (this != &other)
	{
		iconnector::operator = (other);
	}
	return *this;
}

const_con& const_con::operator = (const_con&& other)
{
	if (this != &other)
	{
		iconnector::operator = (std::move(other));
	}
	return *this;
}

std::unordered_set<ileaf*> const_con::get_leaves (void) const
{
	return get_arguments()[0]->get_leaves();
}

tensor* const_con::get_tensor (void)
{
	return get_arguments()[0]->get_tensor();
}

varptr const_con::derive (inode* wrt)
{
	tensor* data = get_tensor();
	if (data)
	{
		tensorshape shape = data->get_shape();
		std::vector<double> zeroes(shape.n_elems(), 0); // todo: convert to data type
		return constant::get(zeroes, shape);
	}
	return nullptr;
}

void const_con::update (void)
{
	this->notify(UPDATE);
}

const_con::const_con (inode* x) :
	iconnector(std::vector<inode*>{x}, "as_const")
{ this->update(); }

const_con::const_con (const const_con& other) : iconnector(other) {}

const_con::const_con (const_con&& other) : iconnector(std::move(other)) {}

inode* const_con::clone_impl (void) const
{
	return new const_con(*this);
}

inode* const_con::move_impl (void)
{
	return new const_con(std::move(*this));
}

}

#endif
