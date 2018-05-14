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

#include "clay/ibuilder.hpp"
#include "mold/constant.hpp"

#pragma once
#ifndef MOLD_VARIABLE_HPP
#define MOLD_VARIABLE_HPP

namespace mold
{

class Variable final : public iNode
{
public:
	bool has_data (void) const override;

    clay::State get_state (void) const override;

    NodePtrT derive (NodeRefT wrt) override;

    bool initialize (const clay::iBuilder& builder);

    bool initialize (const clay::iBuilder& builder, clay::Shape shape);

    void assign (const clay::iSource& src);

private:
    void notify_init (void);

    clay::TensorPtrT data_ = nullptr;
};

}

#endif /* MOLD_VARIABLE_HPP */
