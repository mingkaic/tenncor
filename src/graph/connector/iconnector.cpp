//
//  iconnector.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include <queue>

#include "include/graph/connector/iconnector.hpp"

#ifdef TENNCOR_ICONNECTOR_HPP

namespace nnet
{

inline std::vector<iconnector*> to_con (std::vector<inode*> args)
{
	std::vector<iconnector*> conns;
	for (inode* a : args)
	{
		if (iconnector* con = dynamic_cast<iconnector*>(a))
		{
			conns.push_back(con);
		}
	}
	return conns;
}

iconnector::~iconnector (void){}

iconnector* iconnector::clone (void) const
{
	return static_cast<iconnector*>(this->clone_impl());
}

iconnector* iconnector::move (void)
{
	return static_cast<iconnector*>(this->move_impl());
}

iconnector& iconnector::operator = (const iconnector& other)
{
	if (this != &other)
	{
		iobserver::operator = (other);
		inode::operator = (other);
		copy_helper(other);
	}
	return *this;
}

iconnector& iconnector::operator = (iconnector&& other)
{
	if (this != &other)
	{
		iobserver::operator = (std::move(other));
		inode::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}


std::string iconnector::get_name (void) const
{
	std::string args = "";
	auto it = this->dependencies_.begin();
	auto et = this->dependencies_.end();
	const inode * arg = dynamic_cast<const inode*>(*it);
	while (args.empty() && nullptr == arg)
	{
		arg = dynamic_cast<const inode*>(*++it);
	}
	if (arg)
	{
		args = arg->get_label();
		++it;
	}
	while (it != et)
	{
		if (nullptr != (arg = dynamic_cast<const inode*>(*it)))
		{
			args += "," + arg->get_label();
		}
		it++;
	}
	return inode::get_name() + "(" + args + ")";
}

size_t iconnector::get_depth (void) const
{
	return depth_;
}

std::vector<inode*> iconnector::get_arguments (void) const
{
	std::vector<inode*> node_args(this->dependencies_.size());
	std::transform(this->dependencies_.begin(), this->dependencies_.end(), node_args.begin(),
		[](subject* s) { return static_cast<inode*>(s); });
	return node_args;
}


iconnector::iconnector (std::vector<inode*> dependencies, std::string label) :
	inode(label), iobserver(std::vector<subject*>(dependencies.begin(), dependencies.end()))
{
	size_t ndeps = dependencies.size();
	if (ndeps > 0)
	{
		std::vector<size_t> depths(ndeps, 0);
		std::transform(dependencies.begin(), dependencies.end(), depths.begin(),
		[](inode* node)
		{
			return node->get_depth();
		});
		depth_ = *(std::max_element(depths.begin(), depths.end())) + 1;
	}
}

iconnector::iconnector (const iconnector& other) :
	inode(other), iobserver(other)
{
	copy_helper(other);
}

iconnector::iconnector (iconnector&& other) :
	inode(std::move(other)), iobserver(std::move(other))
{
	move_helper(std::move(other));
}


void iconnector::copy_helper (const iconnector& other)
{
	depth_ = other.depth_;
}

void iconnector::move_helper (iconnector&& other)
{
	depth_ = std::move(other.depth_);
}

}

#endif
