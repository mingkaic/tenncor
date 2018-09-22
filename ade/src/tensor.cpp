#include "ade/functor.hpp"

#ifdef ADE_TENSOR_HPP

namespace ade
{

Tensorptr Tensor::SYMBOLIC_ONE = new Tensor{Shape()};

Tensorptr Tensor::SYMBOLIC_ZERO = new Tensor{Shape()};

Tensorptr constant_one (std::vector<DimT> shape)
{
	auto out = Functor<RESHAPE,std::vector<DimT>>::get(
		{Tensor::SYMBOLIC_ONE}, shape);
	return out;
}

Tensorptr constant_zero (std::vector<DimT> shape)
{
	auto out = Functor<RESHAPE,std::vector<DimT>>::get(
		{Tensor::SYMBOLIC_ZERO}, shape);
	return out;
}

}

#endif
