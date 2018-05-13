/*!
 *
 *  constant.hpp
 *  mold
 *
 *  Purpose:
 *  immutable implementation of inode
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/inode.hpp"

#pragma once
#ifndef MOLD_CONSTANT_HPP
#define MOLD_CONSTANT_HPP

namespace mold
{

class Constant final : public iNode
{
public:
    clay::State get_data (void) const override
    {
        return state_;
    }

    NodePtrT derive (NodeRefT wrt) override;

    void notify (MSG msg) const override;

private:
    clay::State state_;

    std::shared_ptr<char> data_;
};

}

#endif /* MOLD_CONSTANT_HPP */
