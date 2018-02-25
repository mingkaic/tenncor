//
//  placeholder.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/placeholder.hpp"

#ifdef TENNCOR_PLACEHOLDER_HPP

namespace nnet
{

placeholder::placeholder (const tensorshape& shape, std::string name) :
	ileaf(name), data_(new tensor(shape)) {}

placeholder::placeholder (const placeholder& other) :
	ileaf(other) {}

placeholder::placeholder (placeholder&& other) :
	ileaf(std::move(other)) {}

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
		ileaf::operator = (other);
		copy_helper(other);
		this->notify(UPDATE);
	}
	return *this;
}

placeholder& placeholder::operator = (placeholder&& other)
{
	if (this != &other)
	{
		ileaf::operator = (std::move(other));
		move_helper(std::move(other));
		this->notify(UPDATE);
	}
	return *this;
}


tensor* placeholder::get_tensor (void)
{
	return data_.get();
}

varptr placeholder::derive (inode*)
{
	return nullptr;
}

// changes shape
placeholder& placeholder::operator = (tensor& data)
{
	if (&data != data_.get())
	{
		data_->write_to(asgn_);
		data_->read_from(asgn_);
		this->notify(UPDATE);
	}
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

void placeholder::copy_helper (const placeholder& other)
{
	if (nullptr != other.data_)
	{
		data_ = std::make_unique<tensor>(*other.data_);
	}
	else
	{
		data_ = nullptr;
	}
}

void placeholder::move_helper (placeholder&& other)
{
	data_ = std::move(other.data_);
}

placeptr::placeptr (placeholder* ptr) : varptr(ptr) {}

placeptr& placeptr::operator = (placeholder* other)
{
	varptr::operator = (other);
	return *this;
}

placeptr& placeptr::operator = (tensor& ten)
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
