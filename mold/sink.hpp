/*!
 *
 *  functor.hpp
 *  mold
 *
 *  Purpose:
 *  functor implementation of inode
 *  performs forward operations when necessary
 *  create gradient nodes when called
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <iostream>

#include "mold/inode.hpp"
#include "mold/iobserver.hpp"

#pragma once
#ifndef MOLD_SINK_HPP
#define MOLD_SINK_HPP

namespace mold
{

class Sink final
{
public:
	Sink (iNode* arg) : death_sink_(new DeathSink(arg, this)) {}

	~Sink (void)
	{
		clear();
	}

	Sink (const Sink& other) : death_sink_(other.death_sink_) {}

	Sink (Sink&& other) : death_sink_(std::move(other.death_sink_)) {}

	Sink& operator = (const Sink& other)
	{
		if (this != &other)
		{
			clear();
			death_sink_ = new DeathSink(*(other.death_sink_), this);
		}
		return *this;
	}

	Sink& operator = (Sink&& other)
	{
		if (this != &other)
		{
			clear();
			death_sink_ = new DeathSink(*(other.death_sink_), this);
		}
		return *this;
	}

	Sink& operator = (iNode* arg)
	{
		clear();
		death_sink_ = new DeathSink(arg, this);
		return *this;
	}

	iNode* get (void) const
	{
		iNode* out = nullptr;
		if (death_sink_ != nullptr)
		{
			out = death_sink_->get();
		}
		return out;
	}

	bool expired (void) const
	{
		return nullptr == death_sink_;
	}

private:
	struct DeathSink final : public iObserver
	{
		DeathSink (iNode* arg, Sink* owner) :
			iObserver({arg}), owner_(owner) {}

		~DeathSink (void)
		{
			owner_->death_sink_ = nullptr;
		}

		DeathSink (const DeathSink& other, Sink* owner) :
			iObserver(other), owner_(owner) {}

		DeathSink (DeathSink&& other, Sink* owner) :
			iObserver(std::move(other)), owner_(owner) {}

		iNode* get (void) const
		{
			return this->args_[0];
		}

		void initialize (void) override {} // todo: add functionality

		void update (void) override {} // todo: add functionality

		Sink* owner_;
	};

	void clear (void)
	{
		if (nullptr == death_sink_)
		{
			delete death_sink_;
		}
	}

	const DeathSink* death_sink_;
};

}

#endif /* MOLD_SINK_HPP */
