//
//  functor.cpp
//  mold
//

#include <cassert>
#include <algorithm>

#include "mold/functor.hpp"
#include "mold/error.hpp"

#ifdef MOLD_FUNCTOR_HPP

namespace mold
{

static std::vector<iNode*> to_nodes (std::vector<NodeRange>& args)
{
	std::vector<iNode*> out;
	std::transform(args.begin(), args.end(), std::back_inserter(out),
	[](NodeRange& nr)
	{
		return nr.arg_;
	});
	return out;
}

static std::vector<Range> to_ranges (std::vector<NodeRange>& args)
{
	std::vector<Range> out;
	std::transform(args.begin(), args.end(), std::back_inserter(out),
	[](NodeRange& nr)
	{
		return nr.drange_;
	});
	return out;
}

Functor::Functor (std::vector<NodeRange> args, OperatePtrT op) :
	iFunctor(to_nodes(args)), ranges_(to_ranges(args)), op_(op)
{
	initialize();
}

Functor::Functor (const Functor& other) :
	iNode(other), iFunctor(other), ranges_(other.ranges_),
	op_(other.op_->clone())
{
	initialize();
}

Functor::Functor (Functor&& other) :
	iNode(std::move(other)), iFunctor(std::move(other)),
	cache_(std::move(other.cache_)), ranges_(std::move(other.ranges_)),
	op_(std::move(other.op_))
{
	initialize();
}

Functor& Functor::operator = (const Functor& other)
{
	if (&other != this)
	{
		iNode::operator = (other);
		iFunctor::operator = (other);
		ranges_ = other.ranges_;
		cache_ = nullptr;
		op_ = std::unique_ptr<iOperateIO>(other.op_->clone());
		initialize();
	}
	return *this;
}

Functor& Functor::operator = (Functor&& other)
{
	if (&other != this)
	{
		iNode::operator = (std::move(other));
		iFunctor::operator = (std::move(other));
		ranges_ = std::move(other.ranges_);
		cache_ = std::move(other.cache_);
		op_ = std::move(other.op_);
		initialize();
	}
	return *this;
}

bool Functor::has_data (void) const
{
	return nullptr != cache_;
}

clay::Shape Functor::get_shape (void) const
{
	return cache_->get_shape();
}

clay::State Functor::get_state (void) const
{
	if (nullptr == cache_)
	{
		throw UninitializedError();
	}
	return cache_->get_state();
}

void Functor::initialize (void)
{
	if (false == std::all_of(args_.begin(), args_.end(),
	[](iNode*& arg)
	{
		return arg->has_data();
	}))
	{
		return;
	}

	cache_ = op_->make_data(get_args());
	for (iObserver* aud : audience_)
	{
		if (iFunctor* f = dynamic_cast<iFunctor*>(aud))
		{
			f->initialize();
		}
	}
}

void Functor::update (void)
{
	if (nullptr != cache_)
	{
		clay::State dest = cache_->get_state();
		if (false == op_->write_data(dest, get_args()))
		{
			throw FunctorUpdateError();
		}
		for (iObserver* aud : audience_)
		{
			if (iFunctor* f = dynamic_cast<iFunctor*>(aud))
			{
				f->initialize();
			}
		}
	}
}

std::vector<Range> Functor::get_ranges (void) const
{
	return ranges_;
}

std::vector<StateRange> Functor::get_args (void) const
{
	size_t n = args_.size();
	assert(n == ranges_.size());
	std::vector<StateRange> args;
	for (size_t i = 0; i < n; ++i)
	{
		args.push_back({args_[i]->get_state(), ranges_[i]});
	}
	return args;
}

iNode* Functor::clone_impl (void) const
{
	return new Functor(*this);
}

}

#endif
