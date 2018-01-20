//
//  placeholder.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/placeholder.hpp"

#ifdef TENNCOR_PLACEHOLDER_HPP

namespace nnet
{

placeholder::placeholder (const tensorshape& shape, 
	tenncor::tensor_proto::tensor_t type, std::string name) :
ivariable(shape, type, nullptr, name) {}

placeholder::placeholder (const placeholder& other) : ivariable(other) {}

placeholder::placeholder (placeholder&& other) : ivariable(std::move(other)) {}

placeholder* placeholder::clone (void) const
{
	return static_cast<placeholder*>(clone_impl());
}

placeholder* placeholder::move (void)
{
	return static_cast<placeholder*>(move_impl());
}

placeholder& placeholder::operator = (const placeholder& other)
{
	if (this != &other)
	{
		ivariable::operator = (other);
	}
	return *this;
}

placeholder& placeholder::operator = (placeholder&& other)
{
	if (this != &other)
	{
		ivariable::operator = (std::move(other));
	}
	return *this;
}

// changes shape
placeholder& placeholder::operator = (itensor& data)
{
	if (this->data_)
	{
		delete this->data_;
	}
	this->data_ = data.move();
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this;
}

inode* placeholder::clone_impl (void) const
{
	return new placeholder(*this);
}

inode* placeholder::move_impl (void)
{
	return new placeholder(std::move(*this));
}

inode* placeholder::get_gradient (variable*)
{
	return constant::get_shared_zero();
}

placeptr::placeptr (placeholder* ptr) : varptr(ptr) {}

placeptr& placeptr::operator = (placeholder* other)
{
	varptr::operator = (other);
	return *this;
}

placeptr& placeptr::operator = (itensor& ten)
{
	*get() = ten;
	return *this;
}

placeptr::operator placeholder* (void) const
{
	return get();
}

placeholder& placeptr::operator * (void)
{
	return *get();
}

placeholder* placeptr::operator -> (void)
{
	return get();
}

placeholder* placeptr::get (void) const
{
	return static_cast<placeholder*>(varptr::get());
}

}

#endif
