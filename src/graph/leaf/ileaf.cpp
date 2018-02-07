//
//  ileaf.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/leaf/ileaf.hpp"

#ifdef TENNCOR_ILEAF_HPP

namespace nnet
{

ileaf::~ileaf (void) {}

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
		this->notify(UPDATE); // content changed
	}
	return *this;
}

ileaf& ileaf::operator = (ileaf&& other)
{
	if (this != &other)
	{
		inode::operator = (other);
		this->notify(UPDATE); // content changed
	}
	return *this;
}


size_t ileaf::get_depth (void) const
{
	return 0; // leaves are 0 distance from the furthest dependent leaf
}

std::unordered_set<ileaf*> ileaf::get_leaves (void) const
{
	return {const_cast<ileaf*>(this)};
}


ileaf::ileaf (std::string name) : inode(name) {}

ileaf::ileaf (const ileaf& other) : inode(other) {}

ileaf::ileaf (ileaf&& other) : inode(std::move(other)) {}



}

#endif