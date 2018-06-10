/*!
 *
 *  inode.hpp
 *  mold
 *
 *  Purpose:
 *  node interface for storing tensor and generating gradient information
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <unordered_set>

#include "clay/tensor.hpp"

#pragma once
#ifndef MOLD_INODE_HPP
#define MOLD_INODE_HPP

namespace mold
{

class iObserver;

using AudienceT = std::unordered_set<iObserver*>;

class iNode
{
public:
	iNode (void) = default;

	virtual ~iNode (void);

	iNode (const iNode&);

	iNode (iNode&& other);

	iNode& operator = (const iNode&);

	iNode& operator = (iNode&& other);

	iNode* clone (void) const;


	virtual bool has_data (void) const = 0;

	virtual clay::Shape get_shape (void) const = 0;

	virtual clay::State get_state (void) const = 0;

	AudienceT get_audience (void) const;

	void add (iObserver* aud);

	void del (iObserver* aud);

protected:
	virtual iNode* clone_impl (void) const = 0;

	AudienceT audience_;
};

}

#endif /* MOLD_INODE_HPP */
