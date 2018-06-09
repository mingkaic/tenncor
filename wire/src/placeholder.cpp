//
//  placeholder.cpp
//  wire
//

#include <cassert>

#include "wire/placeholder.hpp"

#include "mold/variable.hpp"

#ifdef WIRE_PLACEHOLDER_HPP

namespace wire
{

static optional<clay::Shape> guess_shape (clay::Shape shape, size_t limit)
{
	optional<clay::Shape> bestshape;
	// if allowed is fully defined
	if (shape.is_fully_defined())
	{
		if (shape.n_elems() == limit)
		{
			bestshape = shape;
		}
	}
	// if allowed is partially defined
	else if (shape.is_part_defined())
	{
		std::vector<size_t> my_shape = shape.as_list();
		size_t rank = my_shape.size();
		size_t first_undef = my_shape.size();
		size_t known = 1;
		for (size_t i = 0; i < rank; i++)
		{
			if (0 == my_shape[i])
			{
				if (first_undef > i)
				{
					first_undef = i;
				}
				my_shape[i] = 1;
			}
			else
			{
				known *= my_shape[i];
			}
		}
		assert(known > 0);
		if (0 == limit % known)
		{
			my_shape[first_undef] = limit / known;
			bestshape = clay::Shape(my_shape);
		}
	}
	// if allowed is undefined
	else
	{
		bestshape = clay::Shape({limit});
	}
	return bestshape;
}

Placeholder::Placeholder (std::string label, Graph& graph) :
	Identifier(&graph, new mold::Variable(), label) {}

bool Placeholder::init_helper (size_t n,
	clay::Shape shape, clay::DTYPE dtype)
{
	mold::Variable* arg = static_cast<mold::Variable*>(get());
	bool inited = false == arg->has_data();
	if (inited)
	{
		if (false == shape.is_fully_defined())
		{
			optional<clay::Shape> oshape = guess_shape(shape, n);
			if (false == (bool) oshape)
			{
				throw std::logic_error(
					"attempting to assign badly "
					"shaped data to an unallocated tensor");
			}
			shape = *oshape;
		}
		arg->initialize(clay::TensorPtrT(new clay::Tensor(shape, dtype)));
	}
	return inited;
}

}

#endif
