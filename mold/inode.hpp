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

class iNode;

class Functor;

using NodePtrT = std::shared_ptr<iNode>;

using NodeRefT = std::weak_ptr<iNode>;

using FuncPtrT = std::shared_ptr<Functor>;

using FuncRefT = std::weak_ptr<Functor>;

using SourceIdxT = std::unordered_set<size_t>;

using AudienceT = std::unordered_set<Functor*>;

class iNode
{
public:
    virtual ~iNode (void);

    virtual bool has_data (void) const = 0;

    virtual clay::State get_state (void) const = 0;

    virtual NodePtrT derive (NodeRefT wrt) = 0;

    AudienceT get_audience (void) const;

    void add (Functor* aud);

    void del (Functor* aud);

protected:
    AudienceT audience_;
};

}

#endif /* MOLD_INODE_HPP */
