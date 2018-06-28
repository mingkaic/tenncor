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
#include "mold/iobserver.hpp"
#include "mold/ioperate_io.hpp"

#pragma once
#ifndef MOLD_FUNCTOR_HPP
#define MOLD_FUNCTOR_HPP

namespace mold
{

class Functor final : public iNode, public iObserver
{
public:
	Functor (std::vector<DimRange> args, OperatePtrT op);

	Functor (const Functor& other);

	Functor (Functor&& other);

	Functor& operator = (const Functor& other);

	Functor& operator = (Functor&& other);

	bool has_data (void) const override;

	clay::Shape get_shape (void) const override;

	clay::State get_state (void) const override;


	void initialize (void) override;

	void update (void) override;

private:
	std::vector<StateRange> get_args (void) const;

	iNode* clone_impl (void) const override;

	clay::TensorPtrT cache_ = nullptr;

	OperatePtrT op_;
};

}

#endif /* MOLD_FUNCTOR_HPP */
