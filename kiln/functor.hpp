/*!
 *
 *  functor.hpp
 *  kiln
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

#include "kiln/identifier.hpp"

#pragma once
#ifndef KILN_FUNCTOR_HPP
#define KILN_FUNCTOR_HPP

namespace kiln
{

using GradArgsT = std::vector<std::pair<Identifier*,Identifier*>>;

using GradF = std::function<Identifier*(Identifier*,GradArgsT)>;

struct IdRange
{
	Identifier* arg_;
	mold::RangeT drange_;
};

class Functor final : public Identifier
{
public:
	Functor (std::vector<Identifier*> args,
		slip::OPCODE opcode, Graph& graph = Graph::get_global());

	Functor (std::vector<IdRange> args,
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

#endif /* KILN_FUNCTOR_HPP */
