/*!
 *
 *  ondeath.hpp
 *  mold
 *
 *  Purpose:
 *  associates node to function,
 *  trigger function when observed node dies
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/iobserver.hpp"

#pragma once
#ifndef MOLD_ONDEATH_HPP
#define MOLD_ONDEATH_HPP

namespace mold
{

using TermF = std::function<void(void)>;

struct OnDeath final : public iObserver
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

	void initialize (void) override {}

	void update (void) override {}

    void clear_term (void)
    {
        terminate_ = [](){};
    }

private:
	TermF terminate_;
};

}

#endif /* MOLD_ONDEATH_HPP */
