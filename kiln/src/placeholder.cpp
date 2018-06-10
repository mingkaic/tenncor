//
//  placeholder.cpp
//  kiln
//

#include <cassert>

#include "kiln/placeholder.hpp"

#include "ioutil/stream.hpp"

#include "mold/variable.hpp"

#include "slip/error.hpp"

#ifdef KILN_PLACEHOLDER_HPP

namespace kiln
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

bool Placeholder::init_helper (const char* s, size_t n,
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
		clay::Tensor* tens = new clay::Tensor(shape, dtype);
		char* dest = tens->get_state().get();
		std::memcpy(dest, s, n * clay::type_size(dtype));
		arg->initialize(clay::TensorPtrT(tens));
	}
	return inited;
}

void Placeholder::assign_helper (const char* s, size_t n, clay::DTYPE dtype)
{
	mold::iNode* arg = get();
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
	std::memcpy(state.get(), s, n * clay::type_size(dtype));
	mold::AudienceT auds = arg->get_audience();
	for (mold::iObserver* aud : auds)
	{
		aud->update();
	}
}

}

#endif
