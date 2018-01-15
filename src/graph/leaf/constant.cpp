//
//  constant.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/constant.hpp"

#ifdef TENNCOR_CONSTANT_HPP

namespace nnet
{

constant* constant::get_shared_zero (void)
{
	// shared between ALL instances
	static constant shared_zero(0);
	shared_zero.is_managed_ = true;
	return &shared_zero;
}

constant* constant::get_shared_one (void)
{
	// shared between ALL instances
	static constant shared_one(1);
	shared_one.is_managed_ = true;
	return &shared_one;
}

constant* constant::get (double scalar)
{
	return new constant(scalar);
}

constant* constant::get (std::vector<double> raw, tensorshape shape)
{
	return new constant(raw, shape);
}

varptr constant::derive (inode*)
{
	return constant::get_shared_zero();
}

void constant::be_managed (void)
{
	is_managed_ = true;
}

inode* constant::get_gradient (variable*)
{
	return constant::get_shared_zero();
}

constant::constant (double scalar) :
	ileaf(std::vector<size_t>{1},
		nnutils::formatter() << scalar)
{
	const_init<double> init(scalar);
	this->data_->allocate();
	init(*(this->data_));
	this->is_init_ = true;
}

constant::constant (std::vector<double>raw, tensorshape shape) :
	ileaf(shape, raw.empty() ? "<empty>" :
		(nnutils::formatter() << raw.front() << ".." << raw.back()).str())
{
	size_t rawn = raw.size();
	if (false == this->data_->is_alloc())
	{
		// loosely guess fails if n_elems/n_known> raw size
		// we ensure this will never happen by padding with zeros
		if (shape.n_known()> rawn)
		{
			size_t deficiency = shape.n_known() - rawn;
			raw.insert(raw.end(), deficiency, 0);
		}
		optional<tensorshape> propershape = this->data_->loosely_guess_shape(raw);
		assert((bool) propershape);
		this->data_->allocate(*propershape);
	}
	assert(this->data_->is_alloc());
	// we should also pad 0s for well defined shapes
	size_t n = this->data_->n_elems();
	if (n> rawn)
	{
		size_t deficiency = n - rawn;
		raw.insert(raw.end(), deficiency, 0);
	}
	this->assigner_(*(this->data_), raw);
	this->is_init_ = true;
}

void constant::death_on_noparent (void)
{
	if (false == is_managed_ && this->no_audience())
	{
		delete this;
	}
}

inode* constant::clone_impl (void) const
{
	return nullptr;
}

inode* constant::move_impl (void)
{
	return nullptr;
}

}

#endif
