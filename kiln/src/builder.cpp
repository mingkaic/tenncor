//
//  builder.cpp
//  kiln
//

#include "kiln/builder.hpp"

#include "clay/memory.hpp"

#ifdef KILN_BUILDER_HPP

namespace kiln
{

Builder::Builder (Validator validate, clay::DTYPE dtype) :
	dtype_(dtype), validate_(validate) {}

clay::TensorPtrT Builder::get (void) const
{
	clay::TensorPtrT out = nullptr;
	if (validate_.allowed_.is_fully_defined())
	{
		out = get(validate_.allowed_);
	}
	return out;
}

clay::TensorPtrT Builder::get (clay::Shape shape) const
{
	clay::TensorPtrT out = nullptr;
	if (validate_.support(shape, dtype_))
	{
		out = build(shape);
	}
	return out;
}

}

#endif
