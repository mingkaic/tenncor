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

	Identifier* derive (Identifier* wrt) override
    {
        if (false == arg_->has_data())
        {
		    throw std::exception(); // todo: add context
        }
		Identifier* out;
		clay::DTYPE otype = arg_->get_state().dtype_;
		if (this == wrt)
		{
			out = make_one(otype);
		}
		else
		{
			out = make_zero(otype);
		}
		if (nullptr == out)
		{
			throw std::exception(); // todo: add context
		}
		return out;
	}
};

}

#endif /* WIRE_VARIABLE_HPP */
