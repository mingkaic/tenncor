/*!
 *
 *  placeholder.hpp
 *  kiln
 *
 *  Purpose:
 *  variable wrapper with vector assignment
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "kiln/identifier.hpp"

#pragma once
#ifndef KILN_PLACEHOLDER_HPP
#define KILN_PLACEHOLDER_HPP

namespace kiln
{

class Placeholder final : public Identifier
{
public:
	Placeholder (std::string label,
		Graph& graph = Graph::get_global());

	template <typename T>
	bool initialize (std::vector<T> data,
		clay::Shape shape = clay::Shape())
	{
		clay::DTYPE dtype = clay::get_type<T>();
		size_t n = data.size();
		return init_helper((char*) &data[0], n, shape, dtype);
	}

	template <typename T>
	Placeholder& operator = (std::vector<T> data)
	{
		clay::DTYPE dtype = clay::get_type<T>();
		size_t n = data.size();
		assign_helper((char*) &data[0], n, dtype);
		return *this;
	}

private:
	bool init_helper (const char* s, size_t n, clay::Shape shape,
		clay::DTYPE dtype);

	void assign_helper (const char* s, size_t n, clay::DTYPE dtype);
};

}

#endif /* KILN_PLACEHOLDER_HPP */
