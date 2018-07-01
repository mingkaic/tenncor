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
	OnDeath (iNode* arg, TermF term);

	virtual ~OnDeath (void);

	OnDeath (const OnDeath& other, TermF term);

	OnDeath (OnDeath&& other, TermF term);

	OnDeath& operator = (const OnDeath& other) = delete;

	OnDeath& operator = (OnDeath&& other) = delete;

	iNode* get (void) const;

	void initialize (void) override;

	void update (void) override;

	void clear_term (void);

private:
	TermF terminate_;
};

}

#endif /* MOLD_ONDEATH_HPP */
