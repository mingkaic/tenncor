/*!
 *
 *  ondeath.hpp
 *  mold
 *
 *  Purpose:
 *  executes a function when observer node dies
 *
 *  Created by Mingkai Chen on 2018-05-19
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

//! associates node to function,
//! trigger function when observed node dies
struct OnDeath : public iObserver
{
	OnDeath (iNode* arg, TermF term);

	virtual ~OnDeath (void);

	OnDeath (const OnDeath& other) = default;

	OnDeath (OnDeath&& other) = default;

	OnDeath& operator = (const OnDeath& other) = default;

	OnDeath& operator = (OnDeath&& other) = default;

	OnDeath (const OnDeath& other, TermF term);

	OnDeath (OnDeath&& other, TermF term);

	iNode* get (void) const;

	void initialize (void) override {} // todo: add functionality

	void update (void) override {} // todo: add functionality

private:
	TermF terminate_;
};

}

#endif /* MOLD_ONDEATH_HPP */
