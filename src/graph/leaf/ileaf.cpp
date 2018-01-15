//
//  ileaf.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/ileaf.hpp"

#ifdef TENNCOR_ILEAF_HPP

namespace nnet
{

ileaf::~ileaf (void)
{
	delete data_;
}

ileaf* ileaf::clone (void) const
{
	return static_cast<ileaf*>(this->clone_impl());
}

ileaf* ileaf::move (void)
{
	return static_cast<ileaf*>(this->move_impl());
}

ileaf& ileaf::operator = (const ileaf& other)
{
	if (this != &other)
	{
		inode::operator = (other);
		copy_helper(other);
		this->notify(UPDATE); // content changed
	}
	return *this;
}

ileaf& ileaf::operator = (ileaf&& other)
{
	if (this != &other)
	{
		inode::operator = (other);
		move_helper(std::move(other));
		this->notify(UPDATE); // content changed
	}
	return *this;
}

size_t ileaf::get_depth (void) const
{
	return 0; // leaves are 0 distance from the furthest dependent leaf
}

std::vector<inode*> ileaf::get_arguments (void) const
{
	return {};
}

size_t ileaf::n_arguments (void) const
{
	return 0;
}

const tensor<double>* ileaf::eval (void)
{
	return get_eval();
}

tensorshape ileaf::get_shape (void) const
{
	if (nullptr != data_)
	{
		return data_->get_shape();
	}
	return std::vector<size_t>{};
}

std::unordered_set<ileaf*> ileaf::get_leaves (void) const
{
	return {const_cast<ileaf*>(this)};
}

bool ileaf::good_status (void) const
{
	return is_init_;
}

bool ileaf::read_proto (const tenncor::tensor_proto& proto)
{
	bool success = data_->from_proto(proto);
	if (success)
	{
		is_init_ = true;
		this->notify(UPDATE);
	}
	return success;
}

ileaf::ileaf (const tensorshape& shape, std::string name) :
	inode(name),
	data_(new tensor<double>(shape)) {}

ileaf::ileaf (const ileaf& other) :
	inode(other)
{
	copy_helper(other);
}

ileaf::ileaf (ileaf&& other) :
	inode(std::move(other))
{
	move_helper(std::move(other));
}

const tensor<double>* ileaf::get_eval (void) const
{
	if (false == good_status())
	{
		return nullptr;
	}
	return data_;
}

void ileaf::copy_helper (const ileaf& other)
{
	if (data_)
	{
		delete data_;
		data_ = nullptr;
	}
	is_init_ = other.is_init_;
	// copy over data if other has good_status (we want to ignore uninitialized data)
	if (other.data_)
	{
		data_ = new tensor<double>(*other.data_, !other.good_status());
	}
}

void ileaf::move_helper (ileaf&& other)
{
	if (data_)
	{
		delete data_;
	}
	is_init_ = std::move(other.is_init_);
	data_ = std::move(other.data_);
	other.data_ = nullptr;
}

}

#endif