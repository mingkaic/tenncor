//
//  functor.cpp
//  mold
//

#include "mold/functor.hpp"

#ifdef MOLD_FUNCTOR_HPP

namespace mold
{

Functor::Functor (std::vector<iNode*> args, OperateIO fwd, GradF bwd) :
	iObserver(args), fwd_(fwd), bwd_(bwd)
{
	initialize();
}

Functor::Functor (const Functor& other) :
	iNode(other), iObserver(other),
	fwd_(other.fwd_), bwd_(other.bwd_) {}

Functor::Functor (Functor&& other) :
	iNode(std::move(other)), iObserver(std::move(other)),
	cache_(std::move(other.cache_)), fwd_(std::move(other.fwd_)),
	bwd_(std::move(other.bwd_)) {}

Functor& Functor::operator = (const Functor& other)
{
	if (&other != this)
	{
		iNode::operator = (other);
		iObserver::operator = (other);
		cache_ = nullptr;
		fwd_ = other.fwd_;
		bwd_ = other.bwd_;
	}
	return *this;
}

Functor& Functor::operator = (Functor&& other)
{
	if (&other != this)
	{
		iNode::operator = (std::move(other));
		iObserver::operator = (std::move(other));
		cache_ = std::move(other.cache_);
		fwd_ = std::move(other.fwd_);
		bwd_ = std::move(other.bwd_);
	}
	return *this;
}

bool Functor::has_data (void) const
{
	return nullptr == cache_;
}

clay::State Functor::get_state (void) const
{
	if (nullptr == cache_)
	{
		throw std::exception(); // todo: add context
	}
	return cache_->get_state();
}

iNode* Functor::derive (iNode* wrt)
{
	return bwd_(wrt, args_);
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
	
	std::vector<clay::State> inputs(args_.size());
	std::transform(args_.begin(), args_.end(), inputs.begin(),
	[](iNode* arg) -> clay::State
	{
		return arg->get_state();
	});
	fwd_.args_ = inputs;
	cache_ = fwd_.get();

	for (iObserver* aud : audience_)
	{
		aud->initialize();
	}
}

void Functor::update (void)
{
	if (nullptr != cache_)
	{
		if (false == cache_->read_from(fwd_))
		{
			throw std::exception(); // todo: add context
		}
		for (iObserver* aud : audience_)
		{
			aud->update();
		}
	}
}

}

#endif
