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

struct IdRange
{
	UIDRange get_uid (void) const;

	Identifier* arg_;
	mold::Range drange_;
};

using GradArgsT = std::vector<std::pair<IdRange,IdRange>>;

using GradF = std::function<Identifier*(Identifier*,GradArgsT)>;

class Functor final : public Identifier
{
public:
	Functor (std::vector<Identifier*> args,
		slip::OPCODE opcode, Graph& graph = Graph::get_global());

	Functor (std::vector<IdRange> args,
		slip::OPCODE opcode, Graph& graph = Graph::get_global());

	~Functor (void);

	std::vector<UIDRange> get_args (void) const override;

	slip::OPCODE opcode_;

private:
	std::vector<UID> arg_ids_;
};

}

#endif /* KILN_FUNCTOR_HPP */
