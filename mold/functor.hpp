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

#pragma once
#ifndef MOLD_FUNCTOR_HPP
#define MOLD_FUNCTOR_HPP

namespace mold
{

using OperF = std::function<void(State&, std::vector<State>)>;

using GradF = std::function<NodePtrT(NodeRefT, std::vector<NodeRefT>)>;

class Functor final : public iNode, public clay::iSource
{
public:
    clay::State get_data (void) const override
    {
        if (nullptr == cache_)
        {
            throw std::exception();
        }
        return cache_->get_state();
    }

    NodePtrT derive (NodeRefT wrt) override;

    void notify (MSG msg) const override
    {
        switch (msg)
        {
            case DELETE:
                // release all arguments
            break;
	        case UPDATE:
            {
                State out = get_data();
                for (auto aud : audience_)
                {
                    aud->first()->update(out, aud->second());
                }
            }
            break;
        }
    }

    void update (State state, SourceIdxT srcs)
    {
        size_t nargs = args_.size();
        std::vector<State> input(nargs);
        assert(srcidx < nargs);
        for (size_t src : srcs)
        {
            input[src] = state;
        }
        for (size_t i = 0; i < nargs; ++i)
        {
            if (srcs.end() == srcs.find(i))
            {
                input[i] = args_[i]->get_data();
            }
        }
        State out;
        if (nullptr != cache_)
        {
            out = cache_->get_state();
        }
        else
        {
            // todo: implement
        }
        fwd_(out, input);
        for (auto aud : audience_)
        {
            aud->first()->update(out, aud->second());
        }
    }

    std::vector<NodeRef> get_args (void) const;
    {
        return args_;
    }

private:
    // cache_ is disabled for unary functors with only one observer
    optional<TensorPtrT> cache_;

    std::vector<NodeRef> args_;

    OperF fwd_;

    GradF bwd_;
};

}

#endif /* MOLD_FUNCTOR_HPP */
