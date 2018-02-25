//
//  variable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-27.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/variable.hpp"

#ifdef TENNCOR_VARIABLE_HPP

namespace nnet
{

variable::variable (const tensorshape& shape,
	std::shared_ptr<idata_src> source,
	std::string name) :
ileaf(name), src_(source), data_(new tensor(shape)) {}

variable::variable (const variable& other) :
	ileaf(other)
{
	copy_helper(other);
}

variable::variable (variable&& other) :
	ileaf(std::move(other))
{
	move_helper(std::move(other));
}

variable* variable::clone (void) const
{
	return static_cast<variable*>(this->clone_impl());
}

variable* variable::move (void)
{
	return static_cast<variable*>(this->move_impl());
}

variable& variable::operator = (const variable& other)
{
	if (this != &other)
	{
		ileaf::operator = (other);
		copy_helper(other);
		this->notify(UPDATE);
	}
	return *this;
}

variable& variable::operator = (variable&& other)
{
	if (this != &other)
	{
		ileaf::operator = (std::move(other));
		move_helper(std::move(other));
		this->notify(UPDATE);
	}
	return *this;
}


tensor* variable::get_tensor (void)
{
	return data_.get();
}

varptr variable::derive (inode* wrt)
{
	constant* out = nullptr;
	if (this == wrt)
	{
		tensorshape shape = data_->get_shape();
		std::vector<double> data(shape.n_elems(), 1); // change to match wrt type
		out = constant::get(data, shape);
	}
	return out;
}

bool variable::initialize (void)
{
	data_->get_shape().assert_is_fully_defined();
	bool success = data_->read_from(*src_);
	if (success)
	{
		this->notify(UPDATE);
	}
	return success;
}

bool variable::initialize (tensorshape shape)
{
	shape.assert_is_fully_defined();
	bool success = data_->read_from(*src_, shape);
	if (success)
	{
		this->notify(UPDATE);
	}
	return success;
}

bool variable::assign (inode* input, bool notify)
{
	bool successful = input != nullptr;
	if (successful)
	{
		tensor* itens = input->get_tensor();
		successful = itens->has_data();
		if (successful)
		{
			itens->write_to(asgn_);
			data_->read_from(asgn_);
			if (notify)
			{
				this->notify(UPDATE);
			}
			asgn_.clear();
		}
	}
	return successful;
}

bool variable::assign_add (inode* input, bool notify)
{
	bool successful = input != nullptr && data_->has_data();
	if (successful)
	{
		tensor* itens = input->get_tensor();
		successful = itens->has_data();
		if (successful)
		{
			asgn_.set_op("add");
			data_->write_to(asgn_, 0);
			itens->write_to(asgn_, 1);
			data_->read_from(asgn_);
			if (notify)
			{
				this->notify(UPDATE);
			}
			asgn_.clear();
		}
	}
	return successful;
}

bool variable::assign_sub (inode* input, bool notify)
{
	bool successful = input != nullptr && data_->has_data();
	if (successful)
	{
		tensor* itens = input->get_tensor();
		successful = itens->has_data();
		if (successful)
		{
			asgn_.set_op("sub");
			data_->write_to(asgn_, 0);
			itens->write_to(asgn_, 1);
			data_->read_from(asgn_);
			if (notify)
			{
				this->notify(UPDATE);
			}
			asgn_.clear();
		}
	}
	return successful;
}


inode* variable::clone_impl (void) const
{
	return new variable(*this);
}

inode* variable::move_impl (void)
{
	return new variable(std::move(*this));
}

void variable::copy_helper (const variable& other)
{
	if (nullptr == other.data_)
	{
		data_ = nullptr;
	}
	else
	{
		data_ = std::make_unique<tensor>(*other.data_);
	}

	if (nullptr == other.src_)
	{
		src_ = nullptr;
	}
	else
	{
		src_ = std::shared_ptr<idata_src>(other.src_->clone());
	}
}

void variable::move_helper (variable&& other)
{
	data_ = std::move(other.data_);
	src_ = std::move(other.src_);
}

}

#endif
