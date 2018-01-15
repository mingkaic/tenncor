//
//  leaf.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/variable.hpp"

#ifdef TENNCOR_VARIABLE_HPP

namespace nnet
{

variable::variable (double scalar, std::string name) :
	ivariable(std::vector<size_t>{1},
		new const_init<double>(scalar), name)
{
	initialize();
}

variable::variable (const tensorshape& shape, std::string name) :
	ivariable(shape, nullptr, name) {}

variable::variable (const tensorshape& shape,
	const initializer<double>& init, std::string name) :
ivariable(shape, init.clone(), name) {}

variable* variable::clone (void) const
{
	return static_cast<variable*>(clone_impl());
}

variable* variable::move (void)
{
	return static_cast<variable*>(move_impl());
}

void variable::set_initializer (const initializer<double>& init)
{
	if (this->init_)
	{
		delete this->init_;
	}
	this->init_ = init.clone();
}

tensor<double>& variable::initialize (void)
{
	assert(nullptr != this->init_);
	// if not alloc, attempt to allocate, throw if fail
	if (false == this->data_->is_alloc() &&
		false == this->data_->allocate())
	{
		throw std::runtime_error(this->get_label() + " data is not allocated");
	}
	initializer<double>* init = static_cast<initializer<double>*>(this->init_);
	(*init)(*(this->data_));
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

tensor<double>& variable::initialize (tensorshape shape)
{
	assert(this->init_ != nullptr);
	if (false == this->data_->allocate(shape))
	{
		std::stringstream ss;
		ss << "shape ";
		print_shape(shape, ss);
		ss << " failed to allocate " << this->get_label();
		throw std::runtime_error(ss.str());
	}
	initializer<double>* init = static_cast<initializer<double>*>(this->init_);
	(*init)(*(this->data_));
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

variable_updater variable::assign (inode* input) const
{
	assert(input);
	if (constant* con = dynamic_cast<constant*>(input))
	{
		std::vector<double>data = expose<double>(con);
		return [this, data](bool notify)
		{
			tensor<double>* out_tens = this->data_;
			this->assigner_(*out_tens, data);
			if (notify)
			{
				this->notify(notification::UPDATE);
			}
		};
	}

	return [this, input](bool notify)
	{
		tensor<double>* out_tens = this->data_;
		const tensor<double>* in_tens = input->eval();
		assert(in_tens);
		this->assigner_(*out_tens, *in_tens);
		if (notify)
		{
			this->notify(notification::UPDATE);
		}
	};
}

variable_updater variable::assign_add (inode* input) const
{
	assert(input);
	if (constant* con = dynamic_cast<constant*>(input))
	{
		if (*con == 0)
		{
			return [](bool) {};
		}
		std::vector<double>data = expose<double>(con);
		return [this, data](bool notify)
		{
			tensor<double>* out_tens = this->data_;
			this->assigner_(*out_tens, data,
			[](const double& e1, const double& e2) { return e1 + e2; });
			if (notify)
			{
				this->notify(notification::UPDATE);
			}
		};
	}

	return [this, input](bool notify)
	{
		tensor<double>* out_tens = this->data_;
		const tensor<double>* in_tens = input->eval();
		assert(in_tens);
		this->assigner_(*out_tens, *in_tens,
		[](const double& e1, const double& e2) { return e1 + e2; });
		if (notify)
		{
			this->notify(notification::UPDATE);
		}
	};
}

variable_updater variable::assign_sub (inode* input) const
{
	assert(input);
	if (constant* con = dynamic_cast<constant*>(input))
	{
		if (*con == (double)0)
		{
			return [](bool) {};
		}
		std::vector<double>data = expose<double>(con);
		return [this, data](bool notify)
		{
			tensor<double>* out_tens = this->data_;
			this->assigner_(*out_tens, data,
			[](const double& e1, const double& e2) { return e1 - e2; });
			if (notify)
			{
				this->notify(notification::UPDATE);
			}
		};
	}

	return [this, input](bool notify)
	{
		tensor<double>* out_tens = this->data_;
		const tensor<double>* in_tens = input->eval();
		assert(in_tens);
		this->assigner_(*out_tens, *in_tens,
		[](const double& e1, const double& e2) { return e1 - e2; });
		if (notify)
		{
			this->notify(notification::UPDATE);
		}
	};
}

inode* variable::clone_impl (void) const
{
	return new variable(*this);
}

inode* variable::move_impl (void)
{
	return new variable(std::move(*this));
}

inode* variable::get_gradient (variable* leaf)
{
	if (this == leaf)
	{
		return constant::get_shared_one();
	}
	return constant::get_shared_zero();
}

}

#endif
