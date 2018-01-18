//
//  ivariable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-27.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/ivariable.hpp"

#ifdef TENNCOR_IVARIABLE_HPP

namespace nnet
{

ivariable::~ivariable (void)
{
	if (nullptr != init_)
	{
		delete init_;
	}
}

ivariable* ivariable::clone (void) const
{
	return static_cast<ivariable*>(this->clone_impl());
}

ivariable* ivariable::move (void)
{
	return static_cast<ivariable*>(this->move_impl());
}

ivariable& ivariable::operator = (const ivariable& other)
{
	if (this != &other)
	{
		ileaf::operator = (other);
		copy_helper(other);
		this->notify(UPDATE);
	}
	return *this;
}

ivariable& ivariable::operator = (ivariable&& other)
{
	if (this != &other)
	{
		ileaf::operator = (std::move(other));
		move_helper(std::move(other));
		this->notify(UPDATE);
	}
	return *this;
}

varptr ivariable::derive (inode* wrt)
{
	if (this == wrt)
	{
		return constant::get_shared_one();
	}
	return constant::get_shared_zero();
}

bool ivariable::can_init (void) const
{
	return init_ != nullptr;
}

ivariable::ivariable (const tensorshape& shape,
	tenncor::tensor_proto::tensor_t type,
	initializer* init, std::string name) :
ileaf(shape, type, name), init_(init) {}

ivariable::ivariable (const ivariable& other) :
	ileaf(other)
{
	copy_helper(other);
}

ivariable::ivariable (ivariable&& other) :
	ileaf(std::move(other))
{
	move_helper(std::move(other));
}

void ivariable::copy_helper (const ivariable& other)
{
	if (init_ == other.init_) return;
	if (nullptr != init_)
	{
		delete init_;
	}
	if (other.init_)
	{
		init_ = other.init_->clone();
	}
	else
	{
		init_ = nullptr;
	}
}

void ivariable::move_helper (ivariable&& other)
{
	if (init_ == other.init_) return;
	if (nullptr != init_)
	{
		delete init_;
	}
	init_ = std::move(other.init_);
	other.init_ = nullptr;
}

}

#endif
