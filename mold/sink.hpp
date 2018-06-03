/*!
 *
 *  sink.hpp
 *  mold
 *
 *  Purpose:
 *  iobserver implementation observing inode
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/ondeath.hpp"

#pragma once
#ifndef MOLD_SINK_HPP
#define MOLD_SINK_HPP

namespace mold
{

class Sink final
{
public:
	Sink (iNode* arg);

	~Sink (void);

	Sink (const Sink& other);

	Sink (Sink&& other);

	Sink& operator = (const Sink& other);

	Sink& operator = (Sink&& other);

	Sink& operator = (iNode* arg);

	iNode* get (void) const;

	bool expired (void) const;

private:
	void copy_helper (const Sink& other);

	void move_helper (Sink&& other);

	void clear (void);

	const OnDeath* death_sink_;
};

}

#endif /* MOLD_SINK_HPP */
