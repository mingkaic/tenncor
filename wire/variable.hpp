/*!
 *
 *  placeholder.hpp
 *  wire
 *
 *  Purpose:
 *  extend variable with vector assignment
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

class Variable : public Identifier
{
public:
	Variable (const clay::iBuilder& builder, std::string label,
		Graph& graph = Graph::get_global()) :
		Identifier(&graph, new mold::Variable(), label,
		[builder](mold::Variable* var)
		{
			var->initialize(builder);
		}) {}

	Variable (const clay::iBuilder& builder, clay::Shape shape,
		std::string label, Graph& graph = Graph::get_global()) :
		Identifier(&graph, new mold::Variable(), label,
		[builder, shape](mold::Variable* var)
		{
			if (shape.is_fully_defined())
			{
				var->initialize(builder, shape);
			}
			else
			{
				var->initialize(builder);
			}
		}) {}
};

}

#endif /* WIRE_VARIABLE_HPP */
