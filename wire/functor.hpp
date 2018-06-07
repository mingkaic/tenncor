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

using GradArgsT = std::vector<std::pair<Identifier*,Identifier*>>;

using GradF = std::function<Identifier*(Identifier*,GradArgsT)>;

class Functor final : public Identifier
{
public:
	Functor (std::vector<Identifier*> args,
		slip::OPCODE opcode, Graph& graph = Graph::get_global());

	~Functor (void);

	std::vector<UID> get_args (void) const override
	{
		return arg_ids_;
	}

	slip::OPCODE opcode_;

private:
	std::vector<UID> arg_ids_;
};

}

#endif /* WIRE_FUNCTOR_HPP */
