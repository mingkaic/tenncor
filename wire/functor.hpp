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

	Identifier* derive (Identifier* wrt) override;

    std::vector<std::string> arg_ids_;

private:
    std::vector<mold::iNode*> to_nodes (std::vector<Identifier*> ids);

    GradF grad_;
};

}

#endif /* WIRE_FUNCTOR_HPP */