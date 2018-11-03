#include "ade/functor.hpp"

#ifdef ADE_TENSOR_HPP

namespace ade
{

Tensorptr Tensor::SYMBOLIC_ONE = new Tensor(Shape());

Tensorptr Tensor::SYMBOLIC_ZERO = new Tensor(Shape());

}

#endif
