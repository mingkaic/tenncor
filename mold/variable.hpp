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

	Variable (const Variable& other)
	{
		if (nullptr != other.data_)
		{
			data_ = clay::TensorPtrT(other.data_->clone());
		}
	}

	Variable& operator = (const Variable& other)
	{
		if (&other != this)
		{
			if (nullptr != other.data_)
			{
				data_ = clay::TensorPtrT(other.data_->clone());
			}
			else
			{
				data_ = nullptr;
			}
		}
		return *this;
	}

	Variable (Variable&&) = default;
	Variable& operator = (Variable&&) = default;

	bool has_data (void) const override;

	clay::Shape get_shape (void) const override
	{
		return data_->get_shape();
	}

	clay::State get_state (void) const override;

	void initialize (clay::TensorPtrT data);

	void assign (const mold::iSource& src);

protected:
	iNode* clone_impl (void) const override
	{
		return new Variable(*this);
	}

private:
	void notify_init (void);

	clay::TensorPtrT data_ = nullptr;
};

}

#endif /* MOLD_VARIABLE_HPP */
