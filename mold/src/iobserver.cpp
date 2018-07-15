//
//  iobserver.cpp
//  mold
//

#include <algorithm>

#include "mold/iobserver.hpp"

#ifdef MOLD_IOBSERVER_HPP

namespace mold
{

iObserver::iObserver (std::vector<iNode*> args) : args_(args)
{
	for (iNode*& arg : args)
	{
		if (nullptr == arg)
		{
			throw std::exception();
		}
		arg->add(this);
	}
}

iObserver::~iObserver (void)
{
	for (iNode* arg : args_)
	{
		arg->del(this);
	}
}

iObserver::iObserver (const iObserver& other)
{
	copy_helper(other);
}

iObserver::iObserver (iObserver&& other)
{
	move_helper(std::move(other));
}

iObserver& iObserver::operator = (const iObserver& other)
{
	if (&other != this)
	{
		copy_helper(other);
	}
	return *this;
}

iObserver& iObserver::operator = (iObserver&& other)
{
	if (&other != this)
	{
		move_helper(std::move(other));
	}
	return *this;
}

std::vector<iNode*> iObserver::get_args (void) const
{
	return args_;
}

void iObserver::copy_helper (const iObserver& other)
{
	for (iNode*& arg : args_)
	{
		arg->del(this);
	}
	args_ = other.args_;
	for (iNode*& arg : args_)
	{
		arg->add(this);
	}
}

void iObserver::move_helper (iObserver&& other)
{
	for (iNode*& arg : args_)
	{
		arg->del(this);
	}
	args_ = std::move(other.args_);
	for (iNode*& arg : args_)
	{
		arg->del(&other);
		arg->add(this);
	}
}

}

#endif
