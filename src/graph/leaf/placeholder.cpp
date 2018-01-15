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

placeholder::placeholder (const tensorshape& shape, std::string name) :
	ivariable(shape, nullptr, name) {}

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

// maintains shape
placeholder& placeholder::operator = (std::vector<double>data)
{
	// note: if this is allocated,
	// compatibility is compared to allocated shape instead of allowed
	assert(this->data_->is_compatible_with(data));

	if (false == this->data_->is_alloc())
	{
		if (optional<tensorshape> cand_shape = this->data_->guess_shape(data))
		{
			this->data_->allocate(*cand_shape);
		}
		// we would reach here if data is empty... (todo: test. currently never reached)
		else
		{
			throw std::logic_error("attempting to assign no data to an unallocated tensor");
		}
	}
	this->assigner_(*(this->data_), data);

	this->is_init_ = true;
	this->notify(UPDATE);
	return *this;
}

// changes shape
placeholder& placeholder::operator = (tensor<double>& data)
{
	*this->data_ = std::move(data);
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

placeptr& placeptr::operator = (std::vector<double>vec)
{
	*get() = vec;
	return *this;
}

placeptr& placeptr::operator = (tensor<double>& ten)
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
