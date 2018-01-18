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

static inline void v_assign_add (void* dest, const void* src, tenncor::tensor_proto::tensor_t type)
{
	switch (type)
	{
		case tenncor::tensor_proto::DOUBLE_T:
			*((double*) dest) += *((const double*) src);
		break;
		case tenncor::tensor_proto::SIGNED_T:
			*((signed*) dest) += *((const signed*) src);
		break;
		default:
		break;
	}
}

static inline void v_assign_sub (void* dest, const void* src, tenncor::tensor_proto::tensor_t type)
{
	switch (type)
	{
		case tenncor::tensor_proto::DOUBLE_T:
			*((double*) dest) -= *((const double*) src);
		break;
		case tenncor::tensor_proto::SIGNED_T:
			*((signed*) dest) -= *((const signed*) src);
		break;
		default:
		break;
	}
}

variable::variable (double scalar, std::string name) :
	ivariable(std::vector<size_t>{1}, 
		tenncor::tensor_proto::DOUBLE_T,
		new const_init(scalar), name)
{
	initialize();
}

variable::variable (const tensorshape& shape, 
	tenncor::tensor_proto::tensor_t type, std::string name) :
	ivariable(shape, type, nullptr, name) {}

variable::variable (const tensorshape& shape, const initializer& init, 
	tenncor::tensor_proto::tensor_t type, std::string name) :
ivariable(shape, type, init.clone(), name) {}

variable* variable::clone (void) const
{
	return static_cast<variable*>(clone_impl());
}

variable* variable::move (void)
{
	return static_cast<variable*>(move_impl());
}

void variable::set_initializer (const initializer& init)
{
	if (this->init_)
	{
		delete this->init_;
	}
	this->init_ = init.clone();
}

itensor& variable::initialize (void)
{
	assert(nullptr != this->init_);
	// if not alloc, attempt to allocate, throw if fail
	if (false == this->data_->is_alloc() &&
		false == this->data_->allocate())
	{
		throw std::runtime_error(this->get_label() + " data is not allocated");
	}
	initializer* init = static_cast<initializer*>(this->init_);
	(*init)(*(this->data_));
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

itensor& variable::initialize (tensorshape shape)
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
	initializer* init = static_cast<initializer*>(this->init_);
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
		const itensor* data = con->eval();
		return [this, data](bool notify)
		{
			this->assigner_(*(this->data_), *data);
			if (notify)
			{
				this->notify(notification::UPDATE);
			}
		};
	}

	return [this, input](bool notify)
	{
		itensor* out_tens = this->data_;
		const itensor* in_tens = input->eval();
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
		const itensor* data = con->eval();
		return [this, data](bool notify)
		{
			itensor* out_tens = this->data_;
			this->assigner_(*out_tens, *data, v_assign_add);
			if (notify)
			{
				this->notify(notification::UPDATE);
			}
		};
	}

	return [this, input](bool notify)
	{
		itensor* out_tens = this->data_;
		const itensor* in_tens = input->eval();
		assert(in_tens);
		this->assigner_(*out_tens, *in_tens, v_assign_add);
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
		const itensor* data = con->eval();
		return [this, data](bool notify)
		{
			itensor* out_tens = this->data_;
			this->assigner_(*out_tens, *data, v_assign_sub);
			if (notify)
			{
				this->notify(notification::UPDATE);
			}
		};
	}

	return [this, input](bool notify)
	{
		itensor* out_tens = this->data_;
		const itensor* in_tens = input->eval();
		assert(in_tens);
		this->assigner_(*out_tens, *in_tens, v_assign_sub);
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
