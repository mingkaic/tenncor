/*!
 *
 *  constant.hpp
 *  wire
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

#include "wire/graph.hpp"
#include "wire/identifier.hpp"

#pragma once
#ifndef WIRE_CONSTANT_HPP
#define WIRE_CONSTANT_HPP

namespace wire
{

struct Constant final : public Identifier
{
	template <typename T>
	static Constant* get (T scalar, Graph& graph = Graph::get_global())
	{
		clay::DTYPE dtype = clay::get_type<T>();
		if (clay::DTYPE::BAD == dtype)
		{
			throw clay::UnsupportedTypeError(dtype);
		}
		std::string label = std::string(ioutil::Stream() << scalar);
		std::shared_ptr<char> ptr = clay::make_char(sizeof(T));
		memcpy(ptr.get(), (char*) &scalar, sizeof(T));
		return new Constant(ptr, std::vector<size_t>{1},
			dtype, label, graph);
	}

	template <typename T>
	static Constant* get (std::vector<T> vec, clay::Shape shape,
		Graph& graph = Graph::get_global())
	{
		clay::DTYPE dtype = clay::get_type<T>();
		if (clay::DTYPE::BAD == dtype)
		{
			throw clay::UnsupportedTypeError(dtype);
		}
		std::string label = (ioutil::Stream() << vec[0] << "...");
		size_t n = vec.size();
		assert(shape.n_elems() == n);
		size_t nbytes = sizeof(T) * n;
		std::shared_ptr<char> ptr = clay::make_char(nbytes);
		memcpy(ptr.get(), (char*) &vec[0], nbytes);
		return new Constant(ptr, shape, dtype,
			label, graph);
	}

	Constant (std::shared_ptr<char> data, clay::Shape shape,
		clay::DTYPE dtype, std::string label,
		Graph& graph = Graph::get_global());
};

Constant* make_zero (Identifier* src);

Constant* make_one (Identifier* src);

}

#endif /* WIRE_CONSTANT_HPP */
