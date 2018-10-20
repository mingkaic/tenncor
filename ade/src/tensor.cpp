#include "ade/functor.hpp"

#ifdef ADE_TENSOR_HPP

namespace ade
{

Tensorptr Tensor::SYMBOLIC_ONE = new Tensor{Shape()};

Tensorptr Tensor::SYMBOLIC_ZERO = new Tensor{Shape()};

Tensorptr constant_one (Shape shape)
{
	auto out = Functor<EXTEND,std::vector<DimT>>::get(
		{Tensor::SYMBOLIC_ONE}, shape.as_list());
	return out;
}

Tensorptr constant_zero (Shape shape)
{
	auto out = Functor<EXTEND,std::vector<DimT>>::get(
		{Tensor::SYMBOLIC_ZERO}, shape.as_list());
	return out;
}

}

#endif
