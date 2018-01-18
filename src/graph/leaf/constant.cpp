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
	static constant shared_zero((double) 0);
	shared_zero.is_managed_ = true;
	return &shared_zero;
}

constant* constant::get_shared_one (void)
{
	// shared between ALL instances
	static constant shared_one((double) 1);
	shared_one.is_managed_ = true;
	return &shared_one;
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
		tenncor::tensor_proto::DOUBLE_T,
		nnutils::formatter() << scalar) 
{ 
	const_init init(scalar); 
	this->data_->allocate(); 
	init(*(this->data_)); 
	this->is_init_ = true; 
} 

constant::constant (std::string name) : ileaf(name) {}

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
