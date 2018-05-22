//
//  sink.cpp
//  mold
//

#include "mold/sink.hpp"
#include "mold/iobserver.hpp"

#ifdef MOLD_SINK_HPP

namespace mold
{

using TermF = std::function<void(void)>;

struct Sink::OnDeath final : public iObserver
{
	OnDeath (iNode* arg, TermF term) :
		iObserver({arg}), terminate_(term) {}

	virtual ~OnDeath (void)
	{
		terminate_();
	}

	OnDeath (const OnDeath& other, TermF term) :
		iObserver(other), terminate_(term) {}

	OnDeath (OnDeath&& other, TermF term) :
		iObserver(std::move(other)), terminate_(std::move(term)) {}

	iNode* get (void) const
	{
		iNode* out = nullptr;
		if (false == this->args_.empty())
		{
			out = this->args_[0];
		}
		return out;
	}

	void initialize (void) override {} // todo: add functionality

	void update (void) override {} // todo: add functionality

private:
	TermF terminate_;
};

Sink::Sink (iNode* arg) : death_sink_(new OnDeath(arg,
	[this]()
	{
		this->death_sink_ = nullptr;
	})) {}

Sink::~Sink (void)
{
	clear();
}

Sink::Sink (const Sink& other)
{
	copy_helper(other);
}

Sink::Sink (Sink&& other)
{
	move_helper(std::move(other));
}

Sink& Sink::operator = (const Sink& other)
{
	if (this != &other)
	{
		clear();
		copy_helper(other);
	}
	return *this;
}

Sink& Sink::operator = (Sink&& other)
{
	if (this != &other)
	{
		clear();
		move_helper(std::move(other));
	}
	return *this;
}

Sink& Sink::operator = (iNode* arg)
{
	clear();
	death_sink_ = new OnDeath(arg,
	[this]()
	{
		this->death_sink_ = nullptr;
	});
	return *this;
}

iNode* Sink::get (void) const
{
	iNode* out = nullptr;
	if (death_sink_ != nullptr)
	{
		out = death_sink_->get();
	}
	return out;
}

bool Sink::expired (void) const
{
	return nullptr == death_sink_;
}

void Sink::copy_helper (const Sink& other)
{
	if (false == other.expired())
	{
		death_sink_ = new OnDeath(*(other.death_sink_),
		[this]()
		{
			this->death_sink_ = nullptr;
		});
	}
	else
	{
		death_sink_ = nullptr;
	}
}

void Sink::move_helper (Sink&& other)
{
	if (false == other.expired())
	{
		death_sink_ = new OnDeath(std::move(*(other.death_sink_)),
		[this]()
		{
			this->death_sink_ = nullptr;
		});
		delete other.death_sink_;
		other.death_sink_ = nullptr;
	}
	else
	{
		death_sink_ = nullptr;
	}
}

void Sink::clear (void)
{
	if (nullptr != death_sink_)
	{
		delete death_sink_;
	}
}

}

#endif
