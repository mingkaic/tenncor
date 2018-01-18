//
//  generator.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-18.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/generator.hpp"

#ifdef ROCNNET_GENERATOR_HPP

namespace nnet
{

generator::~generator (void)
{
	clean_up();
}

generator* generator::get (inode* shape_dep,
	const initializer& init, std::string name)
{
	return new generator(shape_dep, init, name);
}

generator* generator::clone (void) const
{
	return static_cast<generator*>(this->clone_impl());
}

generator* generator::move (void)
{
	return static_cast<generator*>(this->move_impl());
}

generator& generator::operator = (const generator& other)
{
	if (this != &other)
	{
		iconnector::operator = (other);
		clean_up();
		copy_helper(other);
		this->notify(UPDATE);
	}
	return *this;
}

generator& generator::operator = (generator&& other)
{
	if (this != &other)
	{
		iconnector::operator = (std::move(other));
		clean_up();
		move_helper(std::move(other));
		this->notify(UPDATE);
	}
	return *this;
}

void generator::temporary_eval (const iconnector*, inode*& out) const
{
	out = constant::get(1);
}

varptr generator::derive (inode* wrt)
{
	if (this != wrt)
	{
		return constant::get_shared_zero();
	}
	return constant::get_shared_one();
}

tensorshape generator::get_shape (void) const
{
	if (nullptr == data_)
	{
		return tensorshape{};
	}
	return data_->get_shape();
}

std::unordered_set<ileaf*> generator::get_leaves (void) const
{
	return std::unordered_set<ileaf*>{};
}

bool generator::good_status (void) const
{
	return nullptr != data_;
}

bool generator::read_proto (const tenncor::tensor_proto&) { return false; }

void generator::update (std::unordered_set<size_t>)
{
	inode* dep = dynamic_cast<inode*>(this->dependencies_[0]);
	if (nullptr == dep)
	{
		// self destroy
		this->notify(UNSUBSCRIBE);
	}
	tensorshape depshape = dep->get_shape();
	tenncor::tensor_proto::tensor_t deptype = dep->get_type();
	if (false == dep->good_status() || false == depshape.is_fully_defined())
	{
		return;
	}
	if (nullptr == data_)
	{
		// init
		switch (deptype)
		{
			case tenncor::tensor_proto::DOUBLE_T:
				data_ = new tensor_double(depshape);
				return;
			case tenncor::tensor_proto::SIGNED_T:
				data_ = new tensor_signed(depshape);
				return;
			default:
				throw std::exception(); // unsupported type
		}
		(*init_)(*data_);
		this->notify(UPDATE);
	}
	else if (false == data_->get_shape().is_compatible_with(depshape))
	{
		// reshape
		data_->set_shape(depshape);
		(*init_)(*data_);
		this->notify(UPDATE);
	}
	else
	{
		// change shape
	}
}

generator::generator (inode* shape_dep, const initializer& init, std::string name) :
	iconnector({shape_dep}, name)
{
	this->init_ = init.clone();
	this->update(std::unordered_set<size_t>{});
}

generator::generator (const generator& other) :
	iconnector(other)
{
	copy_helper(other);
}

generator::generator (generator&& other) :
	iconnector(std::move(other))
{
	move_helper(std::move(other));
}

inode* generator::clone_impl (void) const
{
	return new generator(*this);
}

inode* generator::move_impl (void)
{
	return new generator(std::move(*this));
}

const itensor* generator::get_eval (void) const
{
	return data_;
}

inode* generator::get_gradient (variable*)
{
	return constant::get_shared_zero();
}

void generator::death_on_broken (void)
{
	delete this;
}

void generator::death_on_noparent (void)
{
	delete this;
}

void generator::copy_helper (const generator& other)
{
	if (data_)
	{
		delete data_;
	}
	if (init_)
	{
		delete init_;
	}

	if (other.init_)
	{
		init_ = other.init_->clone();
	}
	if (other.data_)
	{
		data_ = other.data_->clone();
	}
}

void generator::move_helper (generator&& other)
{
	if (data_)
	{
		delete data_;
	}
	if (init_)
	{
		delete init_;
	}

	if (other.init_)
	{
		init_ = other.init_->move();
	}
	if (other.data_)
	{
		data_ = other.data_;
		other.data_ = nullptr;
	}
}

void generator::clean_up (void)
{
	if (init_) delete init_;
	if (data_) delete data_;
	init_ = nullptr;
	data_ = nullptr;
}

}

#endif
