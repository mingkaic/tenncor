/*!
 *
 *  variable.hpp
 *  mold
 *
 *  Purpose:
 *  variable implementation of inode
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/inode.hpp"

#pragma once
#ifndef MOLD_VARIABLE_HPP
#define MOLD_VARIABLE_HPP

namespace mold
{

class Variable final : public iNode
{
public:
	Variable (void) = default;

	Variable (const Variable& other);

	Variable (Variable&&) = default;

	Variable& operator = (const Variable& other);

	Variable& operator = (Variable&&) = default;

	bool has_data (void) const override;

	clay::Shape get_shape (void) const override;

	clay::State get_state (void) const override;

	void set_data (clay::TensorPtrT data);

private:
	iNode* clone_impl (void) const override;

	clay::TensorPtrT data_ = nullptr;
};

}

#endif /* MOLD_VARIABLE_HPP */
