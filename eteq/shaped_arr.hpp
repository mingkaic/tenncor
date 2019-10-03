#include "teq/shape.hpp"

#include "eteq/generated/pyapi.hpp"

#ifndef ETEQ_SHAPED_ARR_HPP
#define ETEQ_SHAPED_ARR_HPP

namespace eteq
{

template <typename T>
struct ShapedArr final
{
	ShapedArr (void) = default;

	ShapedArr (teq::Shape shape, T data = 0) :
		data_(shape.n_elems(), data), shape_(shape) {}

	std::vector<T> data_;

	teq::Shape shape_;
};

}

#endif // ETEQ_SHAPED_ARR_HPP
