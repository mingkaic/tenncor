//
//  iobserver.cpp
//  mold
//

#include <algorithm>

#include "mold/iobserver.hpp"

#ifdef MOLD_IOBSERVER_HPP

namespace mold
{

iObserver::iObserver (std::vector<iNode*> args) :
	iObserver([](std::vector<iNode*> nodes) -> std::vector<DimRange>
	{
		std::vector<DimRange> out(nodes.size());
		std::transform(nodes.begin(), nodes.end(), out.begin(),
		[](iNode* node)
		{
			return DimRange{node, RangeT{0,0}};
		});
		return out;
	}(args)) {}

iObserver::iObserver (std::vector<DimRange> args) :
	args_(args)
{
	for (DimRange& arg : args_)
	{
		if (nullptr == arg.arg_)
		{
			throw std::exception();
		}
		arg.arg_->add(this);
	}
}

iObserver::~iObserver (void)
{
	for (DimRange& arg : args_)
	{
		arg.arg_->del(this);
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

std::vector<DimRange> iObserver::get_args (void) const
{
	return args_;
}

void iObserver::copy_helper (const iObserver& other)
{
	for (DimRange& arg : args_)
	{
		arg.arg_->del(this);
	}
	args_ = other.args_;
	for (DimRange& arg : args_)
	{
		arg.arg_->add(this);
	}
}

void iObserver::move_helper (iObserver&& other)
{
	for (DimRange& arg : args_)
	{
		arg.arg_->del(this);
	}
	args_ = std::move(other.args_);
	for (DimRange& arg : args_)
	{
		arg.arg_->del(&other);
		arg.arg_->add(this);
	}
}

}

#endif
