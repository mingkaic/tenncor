/*!
 *
 *  constant.hpp
 *  kiln
 *
 *  Purpose:
 *  mold constant wrapper
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include <cassert>

#include "ioutil/stream.hpp"

#include "clay/memory.hpp"
#include "clay/error.hpp"

#include "mold/constant.hpp"
#include "mold/error.hpp"

#include "kiln/graph.hpp"
#include "kiln/identifier.hpp"

#pragma once
#ifndef KILN_CONSTANT_HPP
#define KILN_CONSTANT_HPP

namespace kiln
{

struct Constant final : public Identifier
{
	template <typename T>
	static Constant* get (T scalar, Graph& graph = Graph::get_global());

	template <typename T>
	static Constant* get (std::vector<T> vec, clay::Shape shape,
		Graph& graph = Graph::get_global());

	Constant (std::shared_ptr<char> data, clay::Shape shape,
		clay::DTYPE dtype, std::string label,
		Graph& graph = Graph::get_global());
};

Constant* make_zero (Identifier* src);

Constant* make_one (Identifier* src);

}

#endif /* KILN_CONSTANT_HPP */

#include "kiln/include/constant.ipp"
