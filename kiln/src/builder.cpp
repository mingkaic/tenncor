//
//  builder.cpp
//  kiln
//

#include "clay/memory.hpp"
#include "kiln/builder.hpp"

#ifdef KILN_BUILDER_HPP

namespace kiln
{

Builder::Builder (Validator validate, clay::DTYPE dtype) :
	dtype_(dtype), validate_(validate) {}

clay::Tensor* Builder::get (void) const
{
	clay::Tensor* out = nullptr;
	if (validate_.allowed_.is_fully_defined())
	{
		out = get(validate_.allowed_);
	}
	return out;
}

clay::Tensor* Builder::get (clay::Shape shape) const
{
	if (validate_.support(shape, dtype_))
	{
		size_t nbytes = shape.n_elems() * clay::type_size(dtype_);
		std::shared_ptr<char> data = clay::make_char(nbytes);
		this->init(data.get(), nbytes);
		return new clay::Tensor(data, shape, dtype_);
	}
	return nullptr;
}

}

#endif
