#include "age/runtime/grader.hpp"
#include "llo/data.hpp"

#ifndef _GENERATED_RUNTIME_HPP
#define _GENERATED_RUNTIME_HPP

namespace age
{

template <typename T>
ade::Tensor* data (T scalar, ade::Shape shape)
{
	return llo::get_variable(std::vector<T>(shape.n_elems(),scalar),shape,err::sprintf("%d",scalar));
}

ade::Opcode sum_opcode (void);

ade::Opcode prod_opcode (void);

}

#endif // _GENERATED_RUNTIME_HPP
