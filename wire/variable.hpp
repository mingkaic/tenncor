/*!
 *
 *  placeholder.hpp
 *  wire
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

#include "wire/graph.hpp"
#include "wire/identifier.hpp"

#pragma once
#ifndef WIRE_VARIABLE_HPP
#define WIRE_VARIABLE_HPP

namespace wire
{

struct Variable : public Identifier
{
	Variable (const clay::iBuilder& builder, std::string label,
		Graph& graph = Graph::get_global());

	Variable (const clay::iBuilder& builder, clay::Shape shape,
		std::string label, Graph& graph = Graph::get_global());

	~Variable (void);

	Identifier* derive (Identifier* wrt) override;
};

}

#endif /* WIRE_VARIABLE_HPP */
