//
//  generator.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-07-18.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/generator.hpp"

#ifdef TENNCOR_GENERATOR_HPP

namespace nnet
{

generator* generator::get (inode* shape_dep,
	std::shared_ptr<idata_source> source, std::string name)
{
	return new generator(shape_dep, source, "generator_" + name);
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
		immutable::operator = (other);
		copy_helper(other);
		this->notify(UPDATE);
	}
	return *this;
}

generator& generator::operator = (generator&& other)
{
	if (this != &other)
	{
		immutable::operator = (std::move(other));
		move_helper(std::move(other));
		this->notify(UPDATE);
	}
	return *this;
}


std::unordered_set<ileaf*> generator::get_leaves (void) const
{
	return std::unordered_set<ileaf*>{};
}


generator::generator (inode* shape_dep, 
	std::shared_ptr<idata_source> source,
	std::string name) :
immutable({shape_dep}, name), source_(source)
{
	this->update();
}

generator::generator (const generator& other) :
	immutable(other)
{
	copy_helper(other);
}

generator::generator (generator&& other) :
	immutable(std::move(other))
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

void generator::forward_pass (std::vector<inode*>& args)
{
	tensor* dep = args[0]->get_tensor();
	if (dep && dep->has_data())
	{
		tensorshape depshape = dep->get_shape();
		if (nullptr == data_)
		{
			data_ = std::make_unique<tensor>(depshape, source_);
		}
		
		if (data_->has_data())
		{
			data_->copy();
		}
		else
		{
			data_->read();
		}
	}
}

varptr generator::backward_pass (inode* wrt)
{
	tensorshape shape = this->get_tensor()->get_shape();
	std::vector<double> data(shape.n_elems(),
		(double) (this == wrt));
	return constant::get(data, shape);
}

void generator::death_on_noparent (void)
{
	delete this;
}


void generator::copy_helper (const generator& other)
{
	source_ = std::shared_ptr<idata_source>(other.source_->clone());
}

void generator::move_helper (generator&& other)
{
	source_ = std::move(other.source_);
}

}

#endif

