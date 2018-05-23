//
//  placeholder.cpp
//  wire
//

#include <cassert>

#include "wire/placeholder.hpp"

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
				if (first_undef> i)
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
		assert(known> 0);
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

bool shape_fits (clay::Shape shape, size_t n)
{
	bool compatible = true;
	// perfect fit
	if (shape.is_fully_defined())
	{
		compatible = n == shape.n_elems();
	}
	else
	{
		size_t known = shape.n_known();
		if (0 < known)
		{
			compatible = 0 == n % known;
		}
	}
	return compatible;
}

Placeholder::Placeholder (std::string label,
	Graph& graph) :
	Identifier(&graph, new mold::Variable(), label,
	[](mold::Variable* var, SetBuilderF setter)
	{
		RawBuilder builder;
		if (setter)
		{
			setter(builder);
		}
		var->initialize(builder);
	}) {}

Placeholder::Placeholder (clay::Shape shape, std::string label,
	Graph& graph) :
	Identifier(&graph, new mold::Variable(), label,
	[shape](mold::Variable* var, SetBuilderF setter)
	{
		RawBuilder builder;
		if (setter)
		{
			setter(builder);
		}
		if (shape.is_fully_defined())
		{
			var->initialize(builder, shape);
		}
		else
		{
			var->initialize(builder);
		}
	}) {}

Placeholder::AssignIO::AssignIO (std::string data,
	clay::Shape shape, clay::DTYPE dtype) :
	data_(data), shape_(shape), dtype_(dtype) {}

bool Placeholder::AssignIO::read_data (clay::State& dest) const
{
	bool success = dest.shape_.is_compatible_with(shape_) &&
		dest.dtype_ == dtype_;
	if (success)
	{
		std::memcpy((void*) dest.data_.lock().get(),
			data_.c_str(), data_.size());
	}
	return success;
}

clay::TensorPtrT Placeholder::RawBuilder::get (void) const
{
	clay::Shape shape({limit_});
	size_t nbytes = limit_ * clay::type_size(dtype_);
	std::shared_ptr<char> ptr = clay::make_char(nbytes);
	return std::make_unique<clay::Tensor>(ptr, shape, dtype_);
}

clay::TensorPtrT Placeholder::RawBuilder::get (clay::Shape shape) const
{
	optional<clay::Shape> oshape = guess_shape(shape, limit_);
	if (false == (bool) oshape)
	{
		throw std::logic_error(
			"attempting to assign badly shaped data to an unallocated tensor");
	}
	size_t nbytes = limit_ * clay::type_size(dtype_);
	std::shared_ptr<char> ptr = clay::make_char(nbytes);
	return clay::TensorPtrT(new clay::Tensor(ptr, *oshape, dtype_));
}

clay::iBuilder* Placeholder::RawBuilder::clone_impl (void) const
{
	return new RawBuilder(*this);
}

}

#endif
