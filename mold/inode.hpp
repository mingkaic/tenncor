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

#include "clay/tensor.hpp"

#pragma once
#ifndef MOLD_INODE_HPP
#define MOLD_INODE_HPP

namespace mold
{

class Functor;

using NodePtrT = std::shared_ptr<iNode>;

using NodeRefT = std::weak_ptr<iNode>;

using FuncPtrT = std::shared_ptr<Functor>;

using FuncRefT = std::weak_ptr<Functor>;

using SourceIdxT = std::unordered_set<size_t>;

using AudienceT = std::unordered_map<FuncPtrT, SourceIdxT>;

//! notification messages
enum MSG
{
	DELETE,
	UPDATE
};

class iNode
{
public:
    virtual ~iNode (void) = default;

    virtual clay::State get_data (void) const = 0;

    virtual NodePtrT derive (NodeRefT wrt) = 0;

    virtual void notify (MSG msg) const = 0;

    AudienceT get_audience (void) const
    {
        return audience_;
    }

    void add (FuncPtrT aud, size_t idx)
    {
        audience_[aud].emplace(idx);
    }

    void del (FuncRefT aud, size_t idx)
    {
        audience_[aud].erase(idx);
    }

    void del (FuncRefT aud)
    {
        audience_.erase(aud);
    }

protected:
    AudienceT audience_;
};

}

#endif /* MOLD_INODE_HPP */
