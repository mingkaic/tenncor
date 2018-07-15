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

#include "mold/inode.hpp"
#include "mold/ondeath.hpp"
#include "mold/ioperate_io.hpp"

#pragma once
#ifndef MOLD_FUNCTOR_HPP
#define MOLD_FUNCTOR_HPP

namespace mold
{

struct NodeRange
{
	iNode* arg_;
	Range drange_;
};

struct iFunctor : public iObserver
{
	iFunctor (std::vector<iNode*> args) :
		iObserver(args) {}

	iFunctor (const iFunctor& other) :
		iObserver(other) {}

	iFunctor (iFunctor&& other) :
		iObserver(std::move(other)) {}

	iFunctor& operator = (const iFunctor& other)
	{
		if (&other != this)
		{
			iObserver::operator = (other);
		}
		return *this;
	}

	iFunctor& operator = (iFunctor&& other)
	{
		if (&other != this)
		{
			iObserver::operator = (std::move(other));
		}
		return *this;
	}

	virtual void initialize (void) = 0;

	virtual void update (void) = 0;
};

class Functor final : public iNode, public iFunctor
{
public:
	Functor (std::vector<NodeRange> args, OperatePtrT op);

	Functor (const Functor& other);

	Functor (Functor&& other);

	Functor& operator = (const Functor& other);

	Functor& operator = (Functor&& other);

	bool has_data (void) const override;

	clay::Shape get_shape (void) const override;

	clay::State get_state (void) const override;


	void initialize (void) override;

	void update (void) override;

	std::vector<Range> get_ranges (void) const;

private:
	iNode* clone_impl (void) const override;

	std::vector<StateRange> get_args (void) const;

	clay::TensorPtrT cache_ = nullptr;

	std::vector<Range> ranges_;

	OperatePtrT op_;
};

}

#endif /* MOLD_FUNCTOR_HPP */
