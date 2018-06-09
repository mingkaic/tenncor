/*!
 *
 *  placeholder.hpp
 *  wire
 *
 *  Purpose:
 *  variable wrapper with vector assignment
 *
 *  Created by Mingkai Chen on 2016-11-08
 *  Copyright © 2018 Mingkai Chen. All rights reserved.
 *
 */

#include "wire/identifier.hpp"

#include "ioutil/stream.hpp"

#include "mold/error.hpp"

#include "slip/error.hpp"

#pragma once
#ifndef WIRE_PLACEHOLDER_HPP
#define WIRE_PLACEHOLDER_HPP

namespace wire
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
		bool success = init_helper(data.size(), shape, dtype);
		if (success)
		{
			*this = data;
		}
		return success;
	}

	template <typename T>
	Placeholder& operator = (std::vector<T> data)
	{
		clay::DTYPE dtype = clay::get_type<T>();
		size_t n = data.size();
		mold::Variable* arg = static_cast<mold::Variable*>(get());
		if (false == arg->has_data())
		{
			throw mold::UninitializedError();
		}
		clay::State state = arg->get_state();
		assert(state.shape_.is_fully_defined());
		if (n > state.shape_.n_elems())
		{
			throw std::logic_error(ioutil::Stream() << "data with "
				<< n << " elements cannot be assigned to allcoated tensor with "
				<< state.shape_.n_elems() << " elements");
		}
		if (dtype != state.dtype_)
		{
			throw slip::TypeMismatchError(state.dtype_, dtype);
		}
		std::memcpy(state.data_.lock().get(),
			(char*) &data[0], n * sizeof(T));
		return *this;
	}

private:
	bool init_helper (size_t n,
		clay::Shape shape, clay::DTYPE dtype);
};

}

#endif /* WIRE_PLACEHOLDER_HPP */
