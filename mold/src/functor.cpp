//
//  functor.cpp
//  mold
//

#include <algorithm>

#include "mold/functor.hpp"
#include "mold/error.hpp"

#ifdef MOLD_FUNCTOR_HPP

namespace mold
{

Functor::Functor (std::vector<iNode*> args, OperatePtrT op) :
	iObserver(args), op_(op)
{
	initialize();
}

Functor::Functor (const Functor& other) :
	iNode(other), iObserver(other),
	op_(other.op_->clone())
{
	initialize();
}

Functor::Functor (Functor&& other) :
	iNode(std::move(other)), iObserver(std::move(other)),
	cache_(std::move(other.cache_)), op_(std::move(other.op_))
{
	initialize();
}

Functor& Functor::operator = (const Functor& other)
{
	if (&other != this)
	{
		iNode::operator = (other);
		iObserver::operator = (other);
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
		iObserver::operator = (std::move(other));
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

	std::vector<clay::State> inputs(args_.size());
	std::transform(args_.begin(), args_.end(), inputs.begin(),
	[](iNode* arg) -> clay::State
	{
		return arg->get_state();
	});
	cache_ = op_->make_data(inputs);
	for (iObserver* aud : audience_)
	{
		aud->initialize();
	}
}

void Functor::update (void)
{
	if (nullptr != cache_)
	{
		clay::State dest = cache_->get_state();
		std::vector<clay::State> inputs(args_.size());
		std::transform(args_.begin(), args_.end(), inputs.begin(),
		[](iNode* arg) -> clay::State
		{
			return arg->get_state();
		});
		if (false == op_->write_data(dest, inputs))
		{
			throw FunctorUpdateError();
		}
		for (iObserver* aud : audience_)
		{
			aud->update();
		}
	}
}

}

#endif
