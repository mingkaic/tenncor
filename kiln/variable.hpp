/*!
 *
 *  placeholder.hpp
 *  kiln
 *
 *  Purpose:
 *  mold variable wrapper
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "ioutil/stream.hpp"

#include "mold/variable.hpp"

#include "kiln/graph.hpp"
#include "kiln/identifier.hpp"

#pragma once
#ifndef KILN_VARIABLE_HPP
#define KILN_VARIABLE_HPP

namespace kiln
{

struct Variable final : public Identifier
{
	Variable (const clay::BuildTensorF, std::string label,
		Graph& graph = Graph::get_global());

	~Variable (void);

	bool initialize (clay::TensorPtrT ten);

	bool assign (const Identifier& src);
};

}

#endif /* KILN_VARIABLE_HPP */
