#include "ade/functor.hpp"

#ifdef ADE_TENSOR_HPP

namespace ade
{

Tensorptr Tensor::SYMBOLIC_ONE = new Tensor(Shape());

Tensorptr Tensor::SYMBOLIC_ZERO = new Tensor(Shape());

Tensorptr shaped_one (Shape shape)
{
	return Functor::get(COPY, {{
		extend(0, std::vector<DimT>(shape.begin(), shape.end())),
		Tensor::SYMBOLIC_ONE
	}});
}

Tensorptr shaped_zero (Shape shape)
{
	return Functor::get(COPY, {{
		extend(0, std::vector<DimT>(shape.begin(), shape.end())),
		Tensor::SYMBOLIC_ZERO
	}});
}

}

#endif
