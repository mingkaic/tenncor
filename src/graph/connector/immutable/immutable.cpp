//
//  immutable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-06-26.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/connector/immutable/immutable.hpp"

#ifdef TENNCOR_IMMUTABLE_HPP

namespace nnet
{

immutable::~immutable (void) {}

immutable* immutable::clone (void) const
{
	return static_cast<immutable*>(this->clone_impl());
}

immutable* immutable::move (void)
{
	return static_cast<immutable*>(this->move_impl());
}

immutable& immutable::operator = (const immutable& other)
{
	if (this != &other)
	{
		iconnector::operator = (other);
		copy_helper(other);
	}
	return *this;
}

immutable& immutable::operator = (immutable&& other)
{
	if (this != &other)
	{
		iconnector::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}


std::unordered_set<ileaf*> immutable::get_leaves (void) const
{
	std::unordered_set<ileaf*> leaves;
	std::unordered_set<ileaf*> subleaves;
	std::vector<inode*> args = this->get_arguments();
	for (inode* arg : args)
	{
		subleaves = arg->get_leaves();
		leaves.insert(subleaves.begin(), subleaves.end());
	}
	return leaves;
}

tensor* immutable::get_tensor (void)
{
	return data_.get();
}

varptr immutable::derive (inode* wrt)
{
	if (wrt == this)
	{
		tensor* ten = wrt->get_tensor();
		assert(ten && ten->has_data());
		tensorshape shape = ten->get_shape();
		std::vector<double> data(shape.n_elems(), 1); // change to match wrt type
		return constant::get(data, shape);
	}
	return this->backward_pass(wrt);
}

void immutable::update (void)
{
	std::vector<inode*> args = this->get_arguments();
	bool has_data = std::all_of(args.begin(), args.end(), 
	[](inode* node)
	{
		tensor* tens = node->get_tensor();
		return nullptr != tens && tens->has_data();
	});
	if (has_data)
	{
		forward_pass(args);
		this->notify(UPDATE);
	}
}


immutable::immutable (std::vector<inode*> args, std::string label) :
	iconnector(args, label) {}

immutable::immutable (const immutable& other) :
	iconnector(other)
{
	copy_helper(other);
}

immutable::immutable (immutable&& other) :
	iconnector(std::move(other))
{
	move_helper(std::move(other));
}

void immutable::copy_helper (const immutable& other)
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

void immutable::move_helper (immutable&& other)
{
	data_ = std::move(other.data_);
	other.data_ = nullptr;
}

}

#endif
