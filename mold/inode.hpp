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

struct Range
{
	Range (size_t lower, size_t upper)
	{
		if (lower > upper)
		{
			std::swap(lower, upper);
		}
		lower_ = lower;
		upper_ = upper;
	}

	clay::Shape apply (const clay::Shape& inshape) const
	{
		size_t n = inshape.rank();
		size_t lower = std::min(n - 1, lower_);
		size_t upper = std::min(n, upper_);
		auto bt = inshape.begin();
		if (lower == upper) return {};
		return std::vector<size_t>(bt + lower, bt + upper);
	}

	clay::Shape remove (const clay::Shape& inshape) const
	{
		size_t n = inshape.rank();
		size_t lower = std::min(n - 1, lower_);
		size_t upper = std::min(n, upper_);
		auto bt = inshape.begin();
		if (lower == upper) return inshape;
		std::vector<size_t> out(bt, bt + lower);
		out.insert(out.end(), bt + upper, inshape.end());
		return out;
	}

	size_t lower_;
	size_t upper_;
};

struct StateRange
{
	StateRange (clay::State arg, Range drange) :
		arg_(arg), drange_(drange) {}

	char* get (void) const
	{
		return arg_.get();
	}

	clay::Shape shape (void) const
	{
		return arg_.shape_;
	}

	clay::DTYPE type (void) const
	{
		return arg_.dtype_;
	}

	clay::Shape inner (void) const
	{
		return drange_.apply(arg_.shape_);
	}

	clay::Shape outer (void) const
	{
		return drange_.remove(arg_.shape_);
	}

	clay::State arg_;

	Range drange_;
};

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
