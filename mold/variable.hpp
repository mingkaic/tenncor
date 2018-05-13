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
    clay::State get_data (void) const override
    {
        return data_->get_state();
    }

    NodePtrT derive (NodeRefT wrt) override;

    void notify (MSG msg) const override;

    bool initialize (clay::iBuilder* builder)
    {
        auto out = builder->get();
        bool success = nullptr != out;
        if (success)
        {
            data_ = out;
        }
        return success;
    }

    bool initialize (clay::iBuilder* builder, clay::Shape shape)
    {
        auto out = builder->get(shape);
        bool success = nullptr != out;
        if (success)
        {
            data_ = out;
        }
        return success;
    }

    void assign (clay::iSource* src)
    {
        data_.read_from(src);
    }

private:
    clay::TensorPtrT data_;
};

}

#endif /* MOLD_VARIABLE_HPP */
