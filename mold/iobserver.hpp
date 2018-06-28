/*!
 *
 *  iobserver.hpp
 *  mold
 *
 *  Purpose:
 *  iobserver audience of inode
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/inode.hpp"

#pragma once
#ifndef MOLD_IOBSERVER_HPP
#define MOLD_IOBSERVER_HPP

namespace mold
{

struct DimRange
{
	iNode* arg_;
	Range drange_;
};

// todo: make this pure abstract (move args to functor, add args as iNode to OnDeath)
class iObserver
{
public:
	iObserver (std::vector<iNode*> args); // todo: deprecate this

	iObserver (std::vector<DimRange> args); // todo: move this to functor

	virtual ~iObserver (void);

	iObserver (const iObserver& other);

	iObserver (iObserver&& other);

	iObserver& operator = (const iObserver& other);

	iObserver& operator = (iObserver&& other);

	virtual void initialize (void) = 0;

	virtual void update (void) = 0;

	std::vector<DimRange> get_args (void) const;

protected:
	friend class iNode;

	std::vector<DimRange> args_;

private:
	void copy_helper (const iObserver& other);

	void move_helper (iObserver&& other);
};

}

#endif /* MOLD_IOBSERVER_HPP */
