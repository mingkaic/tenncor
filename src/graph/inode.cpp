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

inode::~inode (void) {}

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

std::string inode::get_summaryid (void) const
{
	return get_name();
}

void inode::set_label (std::string label)
{
	label_ = label;
}

bool inode::find_audience (std::string label, std::unordered_set<inode*>& audience) const
{
	for (auto audpair : audience_)
	{
		iobserver* aud = audpair.first;
		if (inode* anode = dynamic_cast<inode*>(aud))
		{
			if (0 == anode->label_.compare(label))
			{
				audience.insert(anode);
			}
		}
	}
	return false == audience.empty();
}

bool inode::read_proto (const tenncor::tensor_proto& proto)
{
	const tensor* result = get_eval();
	if (result)
	{
		result->from_proto(proto);
	}
	return nullptr != result;
}

TENS_TYPE inode::get_type (void) const
{
	if (const tensor* result = get_eval())
	{
		return result->get_type();
	}
	return BAD;
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

varptr::varptr (inode* ptr) : iobserver({ptr}, false) {}

varptr::~varptr (void) {}

varptr& varptr::operator = (inode* other)
{
	if (this->dependencies_.empty()) this->add_dependency(other);
	else this->replace_dependency(other, 0);
	return *this;
}

varptr::operator inode* (void) const { return get(); }

inode& varptr::operator * (void) const { return *get(); }

inode* varptr::operator -> (void) const { return get(); }

inode* varptr::get (void) const
{
	if (this->dependencies_.empty()) return nullptr;
	return static_cast<inode*>(this->dependencies_.at(0));
}

void varptr::update (std::unordered_set<size_t>) {}

void varptr::clear (void) { this->remove_dependency(0); }

void varptr::death_on_broken (void)
{
	if (false == this->dependencies_.empty())
		this->remove_dependency(0);
}

}

#endif