//
//  inode.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2018 Mingkai Chen. All rights reserved.
//

#include "include/graph/inode.hpp"

#ifdef TENNCOR_INODE_HPP

namespace nnet
{

inode::~inode (void)
{
	graph::get_global().unregister_node(this);
}

inode* inode::clone (void) const
{
	return clone_impl();
}

inode* inode::move (void)
{
	return move_impl();
}

inode& inode::operator = (const inode& other)
{
	if (this != &other)
	{
		subject::operator = (other);
		label_ = other.label_;
	}
	return *this;
}

inode& inode::operator = (inode&& other)
{
	if (this != &other)
	{
		subject::operator = (other);
		label_ = std::move(other.label_);
	}
	return *this;
}

std::string inode::get_uid (void) const
{
	return id_;
}

std::string inode::get_label (void) const
{
	return label_;
}

std::string inode::get_name (void) const
{
	return "<" + label_ + ":" + this->get_uid() + ">";
}

void inode::set_label (std::string label)
{
	label_ = label;
}

inode::inode (std::string label) :
	subject(),
	label_(label) {}

inode::inode (const inode& other) :
	subject(other),
	label_(other.label_) {}

inode::inode (inode&& other) :
	subject(std::move(other)),
	label_(std::move(other.label_)) {}

varptr::varptr (void) : iobserver(false) {}

varptr::varptr (inode* ptr) : iobserver(nullptr == ptr ? 
	std::vector<subject*>{} : std::vector<subject*>{ptr}, false) {}

varptr& varptr::operator = (inode* other)
{
	if (nullptr != other)
	{
		if (this->dependencies_.empty())
		{
			this->add_dependency(other);
		}
		else
		{
			this->replace_dependency(other, 0);
		}
	}
	return *this;
}

varptr::operator inode* (void) const { return get(); }

inode& varptr::operator * (void) const { return *get(); }

inode* varptr::operator -> (void) const { return get(); }

inode* varptr::get (void) const
{
	if (this->dependencies_.empty()) return nullptr;
	return static_cast<inode*>(this->dependencies_.front());
}

void varptr::clear (void) { this->remove_dependency(0); }

void varptr::update (void) {}

void varptr::death_on_broken (void)
{
	this->remove_dependency(0);
}

}

#endif
