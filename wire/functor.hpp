/*!
 *
 *  functor.hpp
 *  wire
 *
 *  Purpose:
 *  mold functor wrapper
 *
 *  Created by Mingkai Chen on 2018-01-12.
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "mold/functor.hpp"

#include "slip/registry.hpp"

#include "wire/identifier.hpp"

#pragma once
#ifndef WIRE_FUNCTOR_HPP
#define WIRE_FUNCTOR_HPP

namespace wire
{

using GradF = std::function<Identifier*(Identifier*, std::vector<Identifier*>)>;

class Functor : public Identifier
{
public:
    Functor (std::vector<Identifier*> args,
        slip::OPCODE opcode, GradF grad,
        Graph& graph = Graph::get_global());

	Identifier* derive (Identifier* wrt) override
    {
        if (false == arg_->has_data())
        {
		    throw std::exception(); // todo: add context
        }
        Identifier* out;
        if (this == wrt)
        {
            out = make_one(arg_->get_state().dtype_);
        }
        else
        {
            std::vector<Identifier*> args(arg_ids_.size());
            std::transform(args.begin(), args.end(), arg_ids_.begin()
            [this](std::string id)
            {
                return graph_->get_node(id);
            });
            out = grad_(wrt, args);
        }
        if (nullptr == out)
        {
            throw std::exception(); // todo: add context
        }
        return out;
    }

    std::vector<std::string> arg_ids_;

private:
    std::vector<mold::iNode*> to_nodes (std::vector<Identifier*> ids);

    GradF grad_;
};

}

#endif /* WIRE_FUNCTOR_HPP */