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
#include "mold/operate_io.hpp"

#pragma once
#ifndef MOLD_FUNCTOR_HPP
#define MOLD_FUNCTOR_HPP

namespace mold
{

using GradF = std::function<NodePtrT(NodeRefT, std::vector<iNode*>)>;

class Functor final : public iNode
{
public:
    Functor (std::vector<iNode*> args, OperateIO fwd, GradF bwd);

	bool has_data (void) const override;

    clay::State get_state (void) const override;

    NodePtrT derive (NodeRefT wrt) override;

    void initialize (void);

    void update (void);

    std::vector<iNode*> get_args (void) const;

private:
    clay::TensorPtrT cache_;

    std::vector<iNode*> args_;

    OperateIO fwd_;

    GradF bwd_;
};

}

#endif /* MOLD_FUNCTOR_HPP */
