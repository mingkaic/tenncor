//
//  inode.cpp
//  mold
//

#include "mold/inode.hpp"
#include "mold/iobserver.hpp"

#ifdef MOLD_INODE_HPP

namespace mold
{

iNode::~iNode (void)
{
	auto auds = audience_;
	for (iObserver* aud : auds)
	{
		if (audience_.end() != audience_.find(aud))
		{
			delete aud;
		}
	}
}

iNode::iNode (const iNode&) {}

iNode::iNode (iNode&& other) :
	audience_(std::move(other.audience_))
{
	for (iObserver* aud : audience_)
	{
		aud->replace(&other, this);
	}
}

iNode& iNode::operator = (const iNode&)
{
	return *this;
}

iNode& iNode::operator = (iNode&& other)
{
	if (&other != this)
	{
		audience_ = std::move(other.audience_);
		for (iObserver* aud : audience_)
		{
			aud->replace(&other, this);
		}
	}
	return *this;
}

AudienceT iNode::get_audience (void) const
{
	return audience_;
}

void iNode::add (iObserver* aud)
{
	audience_.emplace(aud);
}

void iNode::del (iObserver* aud)
{
	audience_.erase(aud);
}

}

#endif
